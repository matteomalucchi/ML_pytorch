import logging
import os
import shutil
import matplotlib

matplotlib.use("Agg")
import argparse
import numpy as np
import mplhep as hep
from scipy import stats
from sklearn.metrics import roc_curve, roc_auc_score, auc, brier_score_loss
from sklearn.calibration import calibration_curve

from hist import Hist
from utils_configs.plot.HEPPlotter import HEPPlotter

from ml_pytorch.utils.lumitext import ENERGY_TEXT, get_lumitext_from_dir

LUMITEXT = ENERGY_TEXT

# CMS colour palette of mplhep: the first colour is used for the background
# and the second one for the signal, consistently in all the plotting scripts
CMS_COLORS = [cycle["color"] for cycle in hep.style.CMS["axes.prop_cycle"]]
BKG_COLOR = CMS_COLORS[0]
SIG_COLOR = CMS_COLORS[1]

# Multi-class only: (name, signal class, signal dataset, [backgrounds]) for
# which the p(signal) / (p(signal) + sum_b p(b)) discriminant is plotted, using
# the output node of the signal class. Only the signal events of the given
# dataset (matched with the short dataset name, see _short_dataset_name) are
# used as signal, or all the events of the class if the dataset is None. The
# class names are matched with the class names of the config, ignoring the case
# and any non-alphanumeric character. Missing classes/datasets are skipped.
SIG_OVER_BKG = [
    ("ggF_SM", "GGF", "c2-0p00-kl-1p00-kt-1p00", ["QCD", "TTbar"]),
    ("VBF_SM", "VBF", "CV-1-C2V-1-C3-1", ["QCD", "TTbar"]),
    ("VBF_k2V0", "VBF", "CV-1-C2V-0-C3-1", ["QCD", "TTbar"]),
]


def get_layout(score_lbl_tensor):
    """Return ``(num_score_cols, label_col, weight_col, kl_col)``.

    The score_lbl_array always stores ``[<scores...>, label, weight, kl]`` so
    the trailing 3 columns are fixed and the number of leading score columns
    depends on whether the model is binary (1 column) or multi-class
    (``num_classes`` columns).
    """
    n_score = score_lbl_tensor.shape[1] - 3
    return n_score, n_score, n_score + 1, n_score + 2


def handle_arrays(score_lbl_tensor, column=0, sig_label=1, bkg_label=0):
    """Backward-compatible binary helper: split into the (signal, background)
    events using the label column derived from the array's layout."""
    _, label_col, _, _ = get_layout(score_lbl_tensor)
    sig = score_lbl_tensor[score_lbl_tensor[:, label_col] == sig_label]
    bkg = score_lbl_tensor[score_lbl_tensor[:, label_col] == bkg_label]

    sig_value = sig[:, column]
    bkg_value = bkg[:, column]

    return sig_value, bkg_value


def _class_label(class_info, class_idx, fallback=None):
    """Lookup a human-readable name for ``class_idx`` from the class metadata
    saved at training time. Falls back to a generic name when missing."""
    if class_info is not None:
        for c in class_info:
            ci = c["class_idx"] if isinstance(c, dict) else c.class_idx
            if int(ci) == int(class_idx):
                name = c["name"] if isinstance(c, dict) else c.name
                lbl = c["lbl"] if isinstance(c, dict) else c.lbl
                return f"{name} (lbl={lbl})"
    return fallback if fallback is not None else f"class {class_idx}"


def my_roc_auc(
    classes: np.ndarray, predictions: np.ndarray, sample_weight: np.ndarray = None
) -> float:
    """
    Calculating ROC AUC score as the probability of correct ordering
    """
    # based on https://github.com/SiLiKhon/my_roc_auc/blob/master/my_roc_auc.py

    if sample_weight is None:
        sample_weight = np.ones_like(predictions)

    assert len(classes) == len(predictions) == len(sample_weight)
    assert classes.ndim == predictions.ndim == sample_weight.ndim == 1
    class0, class1 = sorted(np.unique(classes))

    data = np.empty(
        shape=len(classes),
        dtype=[
            ("c", classes.dtype),
            ("p", predictions.dtype),
            ("w", sample_weight.dtype),
        ],
    )
    data["c"], data["p"], data["w"] = classes, predictions, sample_weight

    data = data[np.argsort(data["c"])]
    data = data[
        np.argsort(data["p"], kind="mergesort")
    ]  # here we're relying on stability as we need class orders preserved

    correction = 0.0
    # mask1 - bool mask to highlight collision areas
    # mask2 - bool mask with collision areas' start points
    mask1 = np.empty(len(data), dtype=bool)
    mask2 = np.empty(len(data), dtype=bool)
    mask1[0] = mask2[-1] = False
    mask1[1:] = data["p"][1:] == data["p"][:-1]
    if mask1.any():
        mask2[:-1] = ~mask1[:-1] & mask1[1:]
        mask1[:-1] |= mask1[1:]
        (ids,) = mask2.nonzero()
        correction = (
            sum(
                [
                    ((dsplit["c"] == class0) * dsplit["w"] * msplit).sum()
                    * ((dsplit["c"] == class1) * dsplit["w"] * msplit).sum()
                    for dsplit, msplit in zip(np.split(data, ids), np.split(mask1, ids))
                ]
            )
            * 0.5
        )

    weights_0 = data["w"] * (data["c"] == class0)
    weights_1 = data["w"] * (data["c"] == class1)
    cumsum_0 = weights_0.cumsum()

    return ((cumsum_0 * weights_1).sum() - correction) / (
        weights_1.sum() * cumsum_0[-1]
    )


def plot_calibration_curve(
    score_lbl_tensor_test,
    dir,
    show,
    comet_logger=None,
    class_info=None,
    n_bins=10,
    kl_bkg_str=None,
    lumitext=LUMITEXT,
):
    """Plot the calibration curves (reliability diagrams) of the classifier.

    * Binary output (one score column): a single curve, with the signal
      (label 1) as the positive class.
    * Multi-class output: one curve per output node, in a one-vs-rest
      fashion. For node ``i`` the true labels are turned into binary ones
      (``1`` for events of class ``i``, ``0`` for all the others).

    Note that ``sklearn.calibration.calibration_curve`` does not support
    event weights, so both the curve and the Brier score are unweighted.

    Produces ``calibration_curve{,_class_<i>}.{png,pdf,svg}`` in ``dir``.
    """
    n_score, label_col, _, _ = get_layout(score_lbl_tensor_test)
    labels = score_lbl_tensor_test[:, label_col].astype(int)

    for out_idx in range(n_score):
        scores = score_lbl_tensor_test[:, out_idx]
        # binary case: the positive class is the signal (label 1), which is
        # stored in the single score column
        ## score vs all
        pos_class = out_idx if n_score > 1 else 1
        binary_labels = (labels == pos_class).astype(int)

        if binary_labels.sum() == 0 or (1 - binary_labels).sum() == 0:
            print(f"Skipping calibration curve for class {pos_class}: only one class present")
            continue

        prob_true, prob_pred = calibration_curve(
            binary_labels, scores, n_bins=n_bins
        )
        brier_score = brier_score_loss(binary_labels, scores)

        if n_score > 1:
            name = _class_label(class_info, out_idx, fallback=f"class {out_idx}")
            base = f"calibration_curve_class_{out_idx}"
            xlabel = f"Mean predicted probability: {name}"
        else:
            name = "Signal"
            base = "calibration_curve"
            xlabel = "Mean predicted probability"
        print(f"Brier score loss ({name}): {brier_score:.4f}")

        series_dict = {
            f"{name} vs rest" if n_score > 1 else "Signal vs background": {
                "data": {"x": [prob_pred, None], "y": [prob_true, None]},
                "style": {"linestyle": "", "fmt": "o", "color": SIG_COLOR},
            },
            "Perfectly calibrated": {
                "data": {"x": [[0, 1], None], "y": [[0, 1], None]},
                "style": {"linestyle": "--", "markersize": 0, "color": "grey"},
            },
        }

        plotter = (
            HEPPlotter("CMS")
            .set_plot_config(figsize=[13, 13], lumitext=lumitext, cmstext="Private")
            .set_output(f"{dir}/{base}")
            .set_labels(
                xlabel=xlabel,
                ylabel="Fraction of positives",
            )
            .set_data(series_dict, plot_type="graph")
            .set_options(
                legend_loc="upper left",
                legend_font_size="small",
                split_legend=False,
                grid=False,
                set_ylim=False,
            )
            .add_annotation(
                x=0.98,
                y=0.05,
                s=f"Brier score loss = {brier_score:.4f}"
                + (f"\n" + rf"bkg $\kappa_\lambda$ = {kl_bkg_str}" if kl_bkg_str else ""),
                fontsize=20,
                ha="right",
                va="bottom",
            )
        )
        if show:
            plotter.show()
        plotter.run()

        if comet_logger:
            comet_logger.log_image(f"{dir}/{base}.png", name=base)

        # =========== score vs single others:
        if n_score > 1:
            for ref_idx in range(n_score):
                if ref_idx == out_idx:
                    continue
                pair_mask = (labels == ref_idx) | (labels == out_idx)
                reduced_labels = (labels[pair_mask] == pos_class).astype(int)
                scores_binary = score_lbl_tensor_test[:, out_idx] / (score_lbl_tensor_test[:, out_idx] + score_lbl_tensor_test[:, ref_idx] )
                reduced_scores = scores_binary[pair_mask]

                if reduced_labels.sum() == 0 or (1 - reduced_labels).sum() == 0:
                    print(
                        f"Skipping calibration curve for class {out_idx} vs {ref_idx}: only one class present"
                    )
                    continue

                prob_true, prob_pred = calibration_curve(
                    reduced_labels, reduced_scores, n_bins=n_bins
                )
                brier_score = brier_score_loss(reduced_labels, reduced_scores)

                name = _class_label(class_info, out_idx, fallback=f"class {out_idx}")
                name_ref = _class_label(class_info, ref_idx, fallback=f"class {ref_idx}")
                base = f"calibration_curve_class_{out_idx}_vs_{ref_idx}"
                xlabel = f"Mean predicted probability: {name}"
                print(f"Brier score loss ({name}): {brier_score:.4f}")

                series_dict = {
                    f"{name} vs {name_ref}" if n_score > 1 else "Signal vs background": {
                        "data": {"x": [prob_pred, None], "y": [prob_true, None]},
                        "style": {"linestyle": "", "fmt": "o", "color": SIG_COLOR},
                    },
                    "Perfectly calibrated": {
                        "data": {"x": [[0, 1], None], "y": [[0, 1], None]},
                        "style": {"linestyle": "--", "markersize": 0, "color": "grey"},
                    },
                }

                plotter = (
                    HEPPlotter("CMS")
                    .set_plot_config(figsize=[13, 13], lumitext=lumitext, cmstext="Private")
                    .set_output(f"{dir}/{base}")
                    .set_labels(
                        xlabel=xlabel,
                        ylabel="Fraction of positives",
                    )
                    .set_data(series_dict, plot_type="graph")
                    .set_options(
                        legend_loc="upper left",
                        legend_font_size="small",
                        split_legend=False,
                        grid=False,
                        set_ylim=False,
                    )
                    .add_annotation(
                        x=0.98,
                        y=0.05,
                        s=f"Brier score loss = {brier_score:.4f}"
                        + (f"\n" + rf"bkg $\kappa_\lambda$ = {kl_bkg_str}" if kl_bkg_str else ""),
                        fontsize=20,
                        ha="right",
                        va="bottom",
                    )
                )
                if show:
                    plotter.show()
                plotter.run()

                if comet_logger:
                    comet_logger.log_image(f"{dir}/{base}.png", name=base)





def weighted_quantile(values, quantile, weights):
    """Weighted quantile: returns the value below which `quantile` fraction of total weight lies."""
    sorted_idx = np.argsort(values)
    sorted_values = values[sorted_idx]
    cumulative_weights = np.cumsum(weights[sorted_idx])
    total_weight = cumulative_weights[-1]
    return float(np.interp(quantile * total_weight, cumulative_weights, sorted_values))


def find_threshold_and_bkg_rejection(
    signal_eff,
    sig_score_test,
    bkg_score_test,
    sig_weight_test,
    bkg_weight_test,
):
    """Find DNN score threshold for target signal efficiency using weighted quantile
    and compute background rejection as weighted fraction below threshold."""
    # (1 - signal_eff) quantile → signal_eff fraction of signal is above threshold
    threshold = weighted_quantile(sig_score_test, 1.0 - signal_eff, sig_weight_test)
    total_bkg_weight = np.sum(bkg_weight_test)
    bkg_rejection = (
        np.sum(bkg_weight_test[bkg_score_test < threshold]) / total_bkg_weight
        if total_bkg_weight > 0
        else 0.0
    )
    return threshold, bkg_rejection


def chi_square(hist_test, hist_train):
    """Chi2/ndof and p-value between the test and the training distribution.

    Both histograms are normalized to unit integral, as they are drawn, and the
    empty bins are skipped. The uncertainties of both are propagated.
    """
    integral_test = hist_test.values().sum()
    integral_train = hist_train.values().sum()
    if integral_test == 0 or integral_train == 0:
        return np.nan, np.nan

    h_test = hist_test.values() / integral_test
    h_train = hist_train.values() / integral_train
    err_test = np.sqrt(hist_test.variances()) / integral_test
    err_train = np.sqrt(hist_train.variances()) / integral_train

    # remove empty bins
    mask = (h_test != 0) & (h_train != 0)

    chi_squared = np.sum(
        (
            (h_test[mask] - h_train[mask])
            / np.sqrt(err_test[mask] ** 2 + err_train[mask] ** 2)
        )
        ** 2
    )
    ndof = len(h_test) - 1

    return chi_squared / ndof, 1 - stats.chi2.cdf(chi_squared, ndof)


def compute_significance(
    dnn_score_target,
    counts_test_list,
    bin_centers,
    bin_width,
    sig_weight_test,
    bkg_weight_test,
    test_fraction,
    rescale,
):
    """Compute significance from binned density histograms above the DNN score threshold.
    Integrates the normalized histograms above the threshold to obtain event fractions,
    then converts to absolute event counts using total weights and rescale factors."""
    bin_index = np.searchsorted(bin_centers, dnn_score_target)
    sig_fraction_above = np.sum(counts_test_list[0][bin_index:] * bin_width[bin_index:])
    bkg_fraction_above = np.sum(counts_test_list[1][bin_index:] * bin_width[bin_index:])
    sig_rescale = rescale[0] if rescale else 1
    bkg_rescale = rescale[1] if rescale else 1
    n_sig_above_target = (
        sig_fraction_above * np.sum(sig_weight_test) / test_fraction * sig_rescale
    )
    n_bkg_above_target = (
        bkg_fraction_above * np.sum(bkg_weight_test) / test_fraction * bkg_rescale
    )
    significance_above_target = np.sqrt(
        2
        * (
            (n_sig_above_target + n_bkg_above_target)
            * np.log(n_sig_above_target / n_bkg_above_target + 1)
            - n_sig_above_target
        )
    )
    return n_sig_above_target, n_bkg_above_target, significance_above_target


def plot_sig_bkg_distributions(
    score_lbl_tensor_train,
    score_lbl_tensor_test,
    dir,
    show,
    rescale,
    test_fraction,
    signal_eff=0.8,
    get_max_significance=False,
    comet_logger=None,
    class_info=None,
    kl_bkg_str=None,
    lumitext=LUMITEXT,
):
    # Dispatch to the multi-class variant when the output has more than one
    # score column (i.e. C >= 3 class probabilities are stored).
    n_score, _, _, _ = get_layout(score_lbl_tensor_test)
    if n_score > 1:
        plot_multiclass_distributions(
            score_lbl_tensor_train,
            score_lbl_tensor_test,
            dir,
            show,
            class_info,
            comet_logger=comet_logger,
            kl_bkg_str=kl_bkg_str,
            lumitext=lumitext,
        )
        return

    # plot the signal and background distributions
    sig_score_train, bkg_score_train = handle_arrays(score_lbl_tensor_train, 0)
    sig_score_test, bkg_score_test = handle_arrays(score_lbl_tensor_test, 0)

    print("sig_score_train", sig_score_train, sig_score_train.shape)
    print("bkg_score_train", bkg_score_train, bkg_score_train.shape)
    print("sig_score_test", sig_score_test, sig_score_test.shape)
    print("bkg_score_test", bkg_score_test, bkg_score_test.shape)

    # get weights
    _, _, weight_col, kl_col = get_layout(score_lbl_tensor_test)
    try:
        sig_weight_train, bkg_weight_train = handle_arrays(
            score_lbl_tensor_train, weight_col
        )
        sig_weight_test, bkg_weight_test = handle_arrays(
            score_lbl_tensor_test, weight_col
        )
    except IndexError:
        print("WARNING: No weights found in the input file. Using equal weights.")
        sig_weight_train = np.ones_like(sig_score_train)
        bkg_weight_train = np.ones_like(bkg_score_train)
        sig_weight_test = np.ones_like(sig_score_test)
        bkg_weight_test = np.ones_like(bkg_score_test)

    print("sig_weight_train", sig_weight_train, sig_weight_train.shape)
    print("bkg_weight_train", bkg_weight_train, bkg_weight_train.shape)
    print("sig_weight_test", sig_weight_test, sig_weight_test.shape)
    print("bkg_weight_test", bkg_weight_test, bkg_weight_test.shape)

    # get the kl values
    try:
        sig_kl_train, bkg_kl_train = handle_arrays(score_lbl_tensor_train, kl_col)
        sig_kl_test, bkg_kl_test = handle_arrays(score_lbl_tensor_test, kl_col)
    except IndexError:
        print("WARNING: No kl values found in the input file. Using equal weights.")
        sig_kl_train = np.ones_like(sig_score_train) * 9999.0
        bkg_kl_train = np.ones_like(bkg_score_train) * 9999.0
        sig_kl_test = np.ones_like(sig_score_test) * 9999.0
        bkg_kl_test = np.ones_like(bkg_score_test) * 9999.0

    print("sig_kl_train", sig_kl_train, sig_kl_train.shape)
    print("bkg_kl_train", bkg_kl_train, bkg_kl_train.shape)
    print("sig_kl_test", sig_kl_test, sig_kl_test.shape)
    print("bkg_kl_test", bkg_kl_test, bkg_kl_test.shape)

    kl_unique_values = list(np.unique(sig_kl_train))
    print("kl_unique_values", kl_unique_values)

    # loop over the differetn kl for signal and take inclusively for bkg
    for kl in kl_unique_values + ["all"]:
        if kl != "all":
            sig_score_train_kl = sig_score_train[sig_kl_train == kl]
            sig_weight_train_kl = sig_weight_train[sig_kl_train == kl]
            sig_score_test_kl = sig_score_test[sig_kl_test == kl]
            sig_weight_test_kl = sig_weight_test[sig_kl_test == kl]
            kl_str = f"{kl:.2f}"
        else:
            sig_score_train_kl = sig_score_train
            sig_weight_train_kl = sig_weight_train
            sig_score_test_kl = sig_score_test
            sig_weight_test_kl = sig_weight_test
            kl_str = "all"

        # HEPPlotter.set_output strips whatever follows the last dot of the
        # output path, so the dots of the kl value are replaced with a "p"
        kl_tag = kl_str.replace("-", "m").replace(".", "p")

        ks_statistic_sig, p_value_sig = stats.ks_2samp(
            sig_score_train_kl, sig_score_test_kl
        )
        ks_statistic_bkg, p_value_bkg = stats.ks_2samp(bkg_score_train, bkg_score_test)
        print(f"\nKS: statistic (sig) = {ks_statistic_sig:.30f}")
        print(f"KS: p-value (sig) = {p_value_sig:.30f}")
        print(f"KS: statistic (bkg) = {ks_statistic_bkg:.30f}")
        print(f"KS: p-value (bkg) = {p_value_bkg:.30f}")

        # Compute significance

        counts_test_list = []
        for score, weight, rescale_factor in zip(
            [sig_score_test_kl, bkg_score_test],
            [sig_weight_test_kl, bkg_weight_test],
            rescale if rescale else [1, 1],
        ):
            counts, bins = np.histogram(
                score,
                weights=weight * rescale_factor,
                bins=1000,
                density=True,
                range=(0, 1),
            )
            counts_test_list.append(counts)
            bin_width = bins[1:] - bins[:-1]
            bin_centers = 0.5 * (bins[:-1] + bins[1:])

        n_sig = (
            np.sum(sig_weight_test_kl) / test_fraction * (rescale[0] if rescale else 1)
        )
        n_bkg = np.sum(bkg_weight_test) / test_fraction * (rescale[1] if rescale else 1)
        significance = n_sig / np.sqrt(n_bkg)
        print(f"\nNumber of signal events in the test dataset: {n_sig}")
        print(f"Number of background events in the test dataset: {n_bkg}")
        print(f"Significance: {significance:.2f}\n")

        lines = []

        if signal_eff != -1:
            if get_max_significance:
                max_significance = -1
                for sig_eff_target in np.linspace(0.0, 1.0, 30):
                    threshold, bkg_rej = find_threshold_and_bkg_rejection(
                        sig_eff_target,
                        sig_score_test_kl,
                        bkg_score_test,
                        sig_weight_test_kl,
                        bkg_weight_test,
                    )
                    n_sig, n_bkg, significance = compute_significance(
                        threshold,
                        counts_test_list,
                        bin_centers,
                        bin_width,
                        sig_weight_test_kl,
                        bkg_weight_test,
                        test_fraction,
                        rescale,
                    )
                    if significance > max_significance:
                        max_significance = significance
                        print("max_significance", max_significance)
                        dnn_score_target = threshold
                        bkg_rejection = bkg_rej
                        n_sig_above_target = n_sig
                        n_bkg_above_target = n_bkg
                        significance_above_target = significance
                        signal_eff = sig_eff_target
            else:
                dnn_score_target, bkg_rejection = find_threshold_and_bkg_rejection(
                    signal_eff,
                    sig_score_test_kl,
                    bkg_score_test,
                    sig_weight_test_kl,
                    bkg_weight_test,
                )
                n_sig_above_target, n_bkg_above_target, significance_above_target = (
                    compute_significance(
                        dnn_score_target,
                        counts_test_list,
                        bin_centers,
                        bin_width,
                        sig_weight_test_kl,
                        bkg_weight_test,
                        test_fraction,
                        rescale,
                    )
                )

            print(
                f"\n###########\nNumber of signal events above {signal_eff:.3f} signal efficiency threshold: {n_sig_above_target:.3f}"
            )
            print(
                f"Number of background events above {signal_eff:.3f} signal efficiency threshold: {n_bkg_above_target:.3f}"
            )
            print(
                f"Significance ({dnn_score_target:.3f} DNN cut): {significance_above_target:.3f}"
            )
            # plot the vertical line for the signal efficiency
            lines.append(
                {
                    "x": dnn_score_target,
                    "color": "grey",
                    "linestyle": "--",
                    "label": "Sig efficiency = {:.2f}\nBkg rejection = {:.2f}\nDNN score cut = {:.2f}".format(
                        signal_eff,
                        bkg_rejection,
                        dnn_score_target,
                    ),
                }
            )

        # Single overtraining plot with signal and background overlaid.
        # The histograms are filled with weights normalized to unit integral,
        # so that the four distributions can be compared with each other.
        def normalized_hist(scores, weights):
            h = Hist.new.Reg(50, 0, 1, name="score").Weight()
            total = np.sum(weights)
            h.fill(scores, weight=weights / total if total else weights)
            return h

        hist_sig_train = normalized_hist(sig_score_train_kl, sig_weight_train_kl)
        hist_sig_test = normalized_hist(sig_score_test_kl, sig_weight_test_kl)
        hist_bkg_train = normalized_hist(bkg_score_train, bkg_weight_train)
        hist_bkg_test = normalized_hist(bkg_score_test, bkg_weight_test)

        series_dict = {
            "Signal (training)": {
                "data": hist_sig_train,
                "style": {
                    "color": SIG_COLOR,
                    "histtype": "step",
                    # "edgecolor": SIG_COLOR,
                    # "facecolor": SIG_COLOR,
                    # "alpha": 0.5,
                },
            },
            "Signal (test)": {
                "data": hist_sig_test,
                "style": {"histtype": "errorbar", "color": SIG_COLOR},
            },
            "Background (training)": {
                "data": hist_bkg_train,
                "style": {"color": BKG_COLOR, "histtype": "step"},
            },
            "Background (test)": {
                "data": hist_bkg_test,
                "style": {"histtype": "errorbar", "color": BKG_COLOR},
            },
        }

        chi2_sig, pvalue_sig = chi_square(hist_sig_test, hist_sig_train)
        chi2_bkg, pvalue_bkg = chi_square(hist_bkg_test, hist_bkg_train)
        print(f"\nchi2/ndof (sig) = {chi2_sig:.3f}, p-value = {pvalue_sig:.3f}")
        print(f"chi2/ndof (bkg) = {chi2_bkg:.3f}, p-value = {pvalue_bkg:.3f}")

        base = f"{dir}/sig_bkg_distributions_kl_{kl_tag}"

        # the histograms are normalized to unit integral by HEPPlotter, so
        # the same log range can be used for every training
        for log in [False, True]:
            plotter = (
                HEPPlotter("CMS")
                .set_plot_config(figsize=[13, 13], lumitext=lumitext, cmstext="Private")
                .set_output(f"{base}{'_log' if log else ''}")
                .set_labels(
                    xlabel="DNN Class Score",
                    ylabel="Normalized counts",
                )
                .set_data(series_dict, plot_type="1d")
                .set_options(
                    legend_loc="upper left",
                    legend_font_size=20,
                    split_legend=False,
                    grid=True,
                    y_log=log,
                    ylim_bottom_value=1e-4 if log else 0.0,
                    ylim_top_value=5 if log else 0.15,
                    # ylim_top_factor=2,
                )
                .add_annotation(
                    x=0.6,
                    y=0.95,
                    s=f"KS sig: p-value = {p_value_sig:.2f}",
                    fontsize=20,
                    ha="left",
                    va="center",
                    color=SIG_COLOR,
                )
                .add_annotation(
                    x=0.6,
                    y=0.9,
                    s=f"KS bkg: p-value = {p_value_bkg:.2f}",
                    fontsize=20,
                    ha="left",
                    va="center",
                    color=BKG_COLOR,
                )
                .add_annotation(
                    x=0.6,
                    y=0.85,
                    s=r"$\chi^2$/ndof= {:.1f},".format(chi2_sig)
                    + f"  p-value= {pvalue_sig:.2f}",
                    fontsize=20,
                    ha="left",
                    va="center",
                    color=SIG_COLOR,
                )
                .add_annotation(
                    x=0.6,
                    y=0.8,
                    s=r"$\chi^2$/ndof= {:.1f},".format(chi2_bkg)
                    + f"  p-value= {pvalue_bkg:.2f}",
                    fontsize=20,
                    ha="left",
                    va="center",
                    color=BKG_COLOR,
                )
                .add_annotation(
                    x=0.6,
                    y=0.75,
                    s=rf"sig $\kappa_\lambda$ = {kl_str}",
                    fontsize=20,
                    ha="left",
                    va="center",
                    color=SIG_COLOR,
                )
                .add_annotation(
                    x=0.6,
                    y=0.7,
                    s=rf"bkg $\kappa_\lambda$ = {kl_bkg_str if kl_bkg_str else 'all'}",
                    fontsize=20,
                    ha="left",
                    va="center",
                    color=BKG_COLOR,
                )
            )
            for line in lines:
                plotter.add_line("v", **line)
            if show:
                plotter.show()
            plotter.run()

            if comet_logger:
                comet_logger.log_image(
                    f"{base}{'_log' if log else ''}.png",
                    name=f"sig_bkg_distributions{'_log' if log else ''}",
                )


def plot_roc_curve(
    score_lbl_tensor_test,
    dir,
    show,
    comet_logger=None,
    class_info=None,
    kl_bkg_str=None,
    lumitext=LUMITEXT,
    sig_over_bkg=SIG_OVER_BKG,
    dataset_idx_test=None,
):
    # Dispatch to multi-class ROCs when more than one score column is stored.
    n_score, label_col, weight_col, kl_col = get_layout(score_lbl_tensor_test)
    if n_score > 1:
        plot_multiclass_sig_over_bkg_roc_curves(
            score_lbl_tensor_test,
            dir,
            show,
            sig_over_bkg,
            class_info=class_info,
            comet_logger=comet_logger,
            kl_bkg_str=kl_bkg_str,
            lumitext=lumitext,
            dataset_idx_test=dataset_idx_test,
        )
        if dataset_idx_test is not None:
            plot_per_dataset_roc_curves(
                score_lbl_tensor_test,
                dataset_idx_test,
                dir,
                show,
                class_info,
                comet_logger=comet_logger,
                kl_bkg_str=kl_bkg_str,
                lumitext=lumitext,
            )
        return

    sig_score_test, bkg_score_test = handle_arrays(score_lbl_tensor_test, 0)
    sig_lbl_test, bkg_lbl_test = handle_arrays(score_lbl_tensor_test, label_col)

    # get the weight
    try:
        sig_weight_test, bkg_weight_test = handle_arrays(
            score_lbl_tensor_test, weight_col
        )
    except IndexError:
        print("WARNING: No weight values found in the input file. Using equal weight.")
        sig_weight_test = np.ones_like(sig_score_test)
        bkg_weight_test = np.ones_like(bkg_score_test)

    print("sig_weight_test", sig_weight_test, sig_weight_test.shape)
    print("bkg_weight_test", bkg_weight_test, bkg_weight_test.shape)

    # get the kl values
    try:
        sig_kl_test, bkg_kl_test = handle_arrays(score_lbl_tensor_test, kl_col)
    except IndexError:
        print("WARNING: No kl values found in the input file. Using equal weights.")
        sig_kl_test = np.ones_like(sig_score_test) * 9999.0
        bkg_kl_test = np.ones_like(bkg_score_test) * 9999.0

    print("sig_kl_test", sig_kl_test, sig_kl_test.shape)
    print("bkg_kl_test", bkg_kl_test, bkg_kl_test.shape)

    kl_unique_values = list(np.unique(sig_kl_test))
    print("kl_unique_values", kl_unique_values)
    roc_info_dict = {}

    # loop over the differetn kl for signal and take inclusively for bkg
    for kl in kl_unique_values + ["all"]:
        if kl != "all":
            sig_score_test_kl = sig_score_test[sig_kl_test == kl]
            sig_weight_test_kl = sig_weight_test[sig_kl_test == kl]
            sig_lbl_test_kl = sig_lbl_test[sig_kl_test == kl]
            kl_str = f"{kl:.2f}"
        else:
            sig_score_test_kl = sig_score_test
            sig_weight_test_kl = sig_weight_test
            sig_lbl_test_kl = sig_lbl_test
            kl_str = "all"

        kl_tag = kl_str.replace("-", "m").replace(".", "p")

        score = np.concatenate((sig_score_test_kl, bkg_score_test))
        weight = np.concatenate((sig_weight_test_kl, bkg_weight_test))
        lbl = np.concatenate((sig_lbl_test_kl, bkg_lbl_test))

        # plot the ROC curve
        fpr, tpr, _ = roc_curve(
            lbl,
            score,
            sample_weight=weight,
        )
        roc_auc = my_roc_auc(
            lbl,
            score,
            sample_weight=weight,
        )

        abs_weights_fpr, abs_weights_tpr, _ = roc_curve(
            lbl,
            score,
            sample_weight=abs(weight),
        )
        abs_weights_roc_auc = roc_auc_score(
            lbl,
            score,
            sample_weight=abs(weight),
        )

        # save tpr and fpr in a npz file

        roc_info_dict[f"tpr_kl_{kl_str}"] = tpr
        roc_info_dict[f"fpr_kl_{kl_str}"] = fpr
        roc_info_dict[f"abs_weights_tpr_kl_{kl_str}"] = abs_weights_tpr
        roc_info_dict[f"abs_weights_fpr_kl_{kl_str}"] = abs_weights_fpr

        # the ROC curves are drawn as graphs without error bars
        series_dict = {
            f"ROC curve - kl = {kl_str} (pos+neg weights AUC = {roc_auc:.3f})": {
                "data": {"x": [tpr, None], "y": [fpr, None]},
                "style": {"linestyle": "-", "markersize": 0},
            },
            f"ROC curve - kl = {kl_str} (abs weights AUC = {abs_weights_roc_auc:.3f})": {
                "data": {"x": [abs_weights_tpr, None], "y": [abs_weights_fpr, None]},
                "style": {"linestyle": "-", "markersize": 0},
            },
        }

        plotter = (
            HEPPlotter("CMS")
            .set_plot_config(lumitext=lumitext)
            .set_output(f"{dir}/roc_curve_kl_{kl_tag}")
            .set_labels(xlabel="True positive rate", ylabel="False positive rate")
            .set_data(series_dict, plot_type="graph")
            .set_options(
                y_log=True,
                legend_loc="upper left",
                legend_font_size="small",
                split_legend=False,
                grid=False,
                set_ylim=False,
            )
            .add_annotation(
                x=0.98,
                y=0.05,
                s=rf"sig $\kappa_\lambda$ = {kl_str}"
                + "\n"
                + rf"bkg $\kappa_\lambda$ = {kl_bkg_str if kl_bkg_str else 'all'}",
                fontsize=16,
                ha="right",
                va="bottom",
            )
        )
        if show:
            plotter.show()
        plotter.run()

        if comet_logger:
            comet_logger.log_image(f"{dir}/roc_curve_kl_{kl_tag}.png", name="roc_curve")

    # save tpr and fpr in a npz file
    np.savez(f"{dir}/tpr_fpr.npz", **roc_info_dict)


def plot_kl_distributions(
    score_lbl_train,
    score_lbl_test,
    out_dir,
    kls_background_to_plot,
    train_test_fraction,
    show=False,
    rescale=None,
    signal_eff=0.8,
    get_max_significance=False,
    do_histos=True,
    do_roc=True,
    do_calibration=True,
    class_info=None,
    comet_logger=None,
    lumitext=LUMITEXT,
    sig_over_bkg=SIG_OVER_BKG,
    dataset_idx_train=None,
    dataset_idx_test=None,
):
    if rescale is None:
        rescale = []

    # The per-background-kl split below indexes the label/kl columns with the
    # binary layout (label at column 1, kl at column 3). For a multi-class
    # output the score block spans several leading columns, so those indices
    # point at class scores instead and the split keeps almost no events (the
    # kl column is also a single dummy value for every non-HH class). Only the
    # inclusive "all" set is meaningful in that case.
    n_score, _, _, _ = get_layout(score_lbl_train)
    if n_score > 1 and list(kls_background_to_plot) != ["all"]:
        print(
            "Multi-class output detected: skipping the per-background-kl split "
            "and producing only the inclusive 'all' plots."
        )
        kls_background_to_plot = ["all"]

    for kl_bkg in kls_background_to_plot:
        kl_bkg_str = "all" if kl_bkg == "all" else f"{kl_bkg:.2f}"

        train_dataset_idx = dataset_idx_train
        test_dataset_idx = dataset_idx_test
        if kl_bkg == "all":
            train_data = score_lbl_train
            test_data = score_lbl_test
        else:
            try:
                sig_train = score_lbl_train[:, 1] == 1
                bkg_train_kl = (score_lbl_train[:, 1] == 0) & (
                    score_lbl_train[:, 3] == float(kl_bkg)
                )
                train_data = score_lbl_train[sig_train | bkg_train_kl]

                sig_test = score_lbl_test[:, 1] == 1
                bkg_test_kl = (score_lbl_test[:, 1] == 0) & (
                    score_lbl_test[:, 3] == float(kl_bkg)
                )
                test_data = score_lbl_test[sig_test | bkg_test_kl]
                if dataset_idx_train is not None:
                    train_dataset_idx = dataset_idx_train[sig_train | bkg_train_kl]
                if dataset_idx_test is not None:
                    test_dataset_idx = dataset_idx_test[sig_test | bkg_test_kl]
            except IndexError:
                train_data = score_lbl_train
                test_data = score_lbl_test

        if do_histos:
            sig_bkg_out_dir = f"{out_dir}/sig_bkg_bkgkl_{kl_bkg_str}"
            os.makedirs(sig_bkg_out_dir, exist_ok=True)
            plot_sig_bkg_distributions(
                train_data,
                test_data,
                sig_bkg_out_dir,
                show,
                rescale,
                train_test_fraction,
                signal_eff=signal_eff,
                get_max_significance=get_max_significance,
                comet_logger=comet_logger,
                class_info=class_info,
                kl_bkg_str=kl_bkg_str,
                lumitext=lumitext,
            )
            if n_score > 1 and sig_over_bkg:
                plot_multiclass_sig_over_bkg(
                    train_data,
                    test_data,
                    sig_bkg_out_dir,
                    show,
                    sig_over_bkg,
                    class_info=class_info,
                    comet_logger=comet_logger,
                    kl_bkg_str=kl_bkg_str,
                    lumitext=lumitext,
                    dataset_idx_train=train_dataset_idx,
                    dataset_idx_test=test_dataset_idx,
                )

        if do_roc:
            roc_out_dir = f"{out_dir}/roc_bkgkl_{kl_bkg_str}"
            os.makedirs(roc_out_dir, exist_ok=True)
            plot_roc_curve(
                test_data,
                roc_out_dir,
                show,
                comet_logger=comet_logger,
                kl_bkg_str=kl_bkg_str,
                lumitext=lumitext,
                class_info=class_info,
                sig_over_bkg=sig_over_bkg,
                dataset_idx_test=test_dataset_idx,
            )

        if do_calibration:
            calib_out_dir = f"{out_dir}/calibration_bkgkl_{kl_bkg_str}"
            os.makedirs(calib_out_dir, exist_ok=True)
            plot_calibration_curve(
                test_data,
                calib_out_dir,
                show,
                comet_logger=comet_logger,
                class_info=class_info,
                kl_bkg_str=kl_bkg_str,
                lumitext=lumitext,
            )


def plot_multiclass_distributions(
    score_lbl_tensor_train,
    score_lbl_tensor_test,
    dir,
    show,
    class_info=None,
    comet_logger=None,
    kl_bkg_str=None,
    lumitext=LUMITEXT,
):
    """Plot, for each output node, the distribution of its score split by true
    class, using :class:`HEPPlotter` so the style matches the binary plots.
    As in the binary case, the training and test distributions of each true
    class are compared with a KS and a chi2 test to check the overtraining.
    Produces ``score_distribution_class_<i>{,_log}.{png,pdf,svg}`` per output."""
    n_score, label_col, weight_col, _ = get_layout(score_lbl_tensor_test)
    train_labels = score_lbl_tensor_train[:, label_col].astype(int)
    test_labels = score_lbl_tensor_test[:, label_col].astype(int)
    train_weights = score_lbl_tensor_train[:, weight_col]
    test_weights = score_lbl_tensor_test[:, weight_col]

    true_classes = sorted(set(np.unique(train_labels)).union(np.unique(test_labels)))
    print(
        f"Multi-class evaluation: output nodes = {n_score}, "
        f"true classes = {true_classes}"
    )

    def normalized_hist(scores, weights):
        h = Hist.new.Reg(30, 0, 1, name="score").Weight()
        total = np.sum(weights)
        h.fill(scores, weight=weights / total if total else weights)
        return h

    for out_idx in range(n_score):
        node_name = _class_label(class_info, out_idx, fallback=f"class {out_idx}")
        train_scores = score_lbl_tensor_train[:, out_idx]
        test_scores = score_lbl_tensor_test[:, out_idx]

        series_dict = {}
        # overtraining tests (KS and chi2 between training and test) of each class
        annotations = []
        for j, true_c in enumerate(true_classes):
            color = CMS_COLORS[j % len(CMS_COLORS)]
            tr_mask = train_labels == true_c
            te_mask = test_labels == true_c
            if tr_mask.sum() == 0 and te_mask.sum() == 0:
                continue

            label = _class_label(class_info, true_c, fallback=f"true class {true_c}")
            hist_train = normalized_hist(train_scores[tr_mask], train_weights[tr_mask])
            hist_test = normalized_hist(test_scores[te_mask], test_weights[te_mask])
            series_dict[f"{label} (training)"] = {
                "data": hist_train,
                "style": {"color": color, "histtype": "step"},
            }
            series_dict[f"{label} (test)"] = {
                "data": hist_test,
                "style": {"color": color, "histtype": "errorbar"},
            }

            if tr_mask.sum() == 0 or te_mask.sum() == 0:
                continue
            _, p_value = stats.ks_2samp(train_scores[tr_mask], test_scores[te_mask])
            chi2, pvalue_chi2 = chi_square(hist_test, hist_train)
            print(
                f"Output node {node_name}, true class {label}: "
                f"KS p-value = {p_value:.3f}, "
                f"chi2/ndof = {chi2:.3f}, p-value = {pvalue_chi2:.3f}"
            )
            annotations.append(
                {
                    "s": f"KS p-value = {p_value:.2f}, "
                    + r"$\chi^2$/ndof= {:.1f},".format(chi2)
                    + f"  p-value= {pvalue_chi2:.2f}",
                    "color": color,
                }
            )

        if kl_bkg_str:
            annotations.append({"s": rf"bkg $\kappa_\lambda$ = {kl_bkg_str}"})

        base = f"{dir}/score_distribution_class_{out_idx}"
        for log in [False, True]:
            plotter = (
                HEPPlotter("CMS")
                .set_plot_config(
                    figsize=[13, 13], lumitext=lumitext, cmstext="Private"
                )
                .set_output(f"{base}{'_log' if log else ''}")
                .set_labels(
                    xlabel=f"Score of output node: {node_name}",
                    ylabel="Normalized counts",
                )
                .set_data(series_dict, plot_type="1d")
                .set_options(
                    legend_loc="upper left",
                    legend_font_size=18,
                    split_legend=False,
                    grid=True,
                    y_log=log,
                    ylim_bottom_value=1e-4 if log else 0.0,
                    # leave room for the legend and the annotations
                    ylim_top_factor=3 if log else 2.2,
                )
            )
            for i, annotation in enumerate(annotations):
                plotter.add_annotation(
                    x=0.98,
                    y=0.95 - 0.045 * i,
                    fontsize=18,
                    ha="right",
                    va="center",
                    **annotation,
                )
            if show:
                plotter.show()
            plotter.run()

            if comet_logger:
                comet_logger.log_image(
                    f"{base}{'_log' if log else ''}.png",
                    name=f"score_distribution_class_{out_idx}"
                    f"{'_log' if log else ''}",
                )


def _save_roc_plot(
    series_dict,
    dir,
    output_base,
    image_name,
    show,
    comet_logger=None,
    lumitext=LUMITEXT,
    annotation=None,
):
    """Draw the ROC curves of ``series_dict`` overlaid on a single figure."""
    if not series_dict:
        return
    plotter = (
        HEPPlotter("CMS")
        .set_plot_config(lumitext=lumitext)
        .set_output(f"{dir}/{output_base}")
        .set_labels(xlabel="True positive rate", ylabel="False positive rate")
        .set_data(series_dict, plot_type="graph")
        .set_options(
            y_log=True,
            legend_loc="upper left",
            legend_font_size="small",
            split_legend=False,
            grid=False,
            set_ylim=False,
        )
    )
    if annotation:
        plotter.add_annotation(
            x=0.98,
            y=0.05,
            s=annotation,
            fontsize=16,
            ha="right",
            va="bottom",
        )
    if show:
        plotter.show()
    plotter.run()
    if comet_logger:
        comet_logger.log_image(f"{dir}/{output_base}.png", name=image_name)


def _normalize_class_name(name):
    """Lowercase and keep only alphanumeric characters, so that e.g.
    ``VBF_k2V0``, ``vbf-k2v0`` and ``VBF k2V=0`` all match each other."""
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _find_class_idx(class_info, name):
    """Return the ``class_idx`` of the class called ``name`` (matched with
    :func:`_normalize_class_name`), or ``None`` if there is no such class."""
    for c in class_info or []:
        c_name = c["name"] if isinstance(c, dict) else c.name
        if _normalize_class_name(c_name) == _normalize_class_name(name):
            return int(c["class_idx"] if isinstance(c, dict) else c.class_idx)
    return None


def _sig_over_bkg_discriminant(score_lbl_tensor, sig_idx, bkg_idxs):
    """D = p(signal) / (p(signal) + sum_b p(b)) from the softmax outputs."""
    sig = score_lbl_tensor[:, sig_idx]
    bkg = score_lbl_tensor[:, bkg_idxs].sum(axis=1)
    den = sig + bkg
    return np.divide(sig, den, out=np.zeros_like(sig), where=den > 0)


def _short_dataset_name(dataset):
    """Compact name of a dataset, e.g.
    ``VBFHHto4B_Par-CV-1-C2V-0-C3-1_TuneCP5_..._2024`` -> ``CV-1-C2V-0-C3-1``."""
    name = dataset.split("_TuneCP5")[0]
    if "_Par-" in name:
        name = name.split("_Par-")[1]
    return name


def _find_dataset_idx(class_info, class_idx, dataset):
    """Return the index of ``dataset`` (matched with the short dataset name)
    within the datasets of class ``class_idx``, or ``None`` if not found."""
    for c in class_info or []:
        if not isinstance(c, dict) or int(c["class_idx"]) != class_idx:
            continue
        for d_idx, d in enumerate(c.get("datasets", [])):
            if _short_dataset_name(d) == dataset:
                return d_idx
    return None


def _resolve_sig_over_bkg(sig_over_bkg, class_info, n_score, dataset_idx=None):
    """Yield ``(name, bkg_names, sig_idx, bkg_idxs, sig_dataset_idx)`` for every
    entry of ``sig_over_bkg`` whose classes (and signal dataset, if given)
    exist and are within the output nodes. ``sig_dataset_idx`` is ``None``
    when the whole signal class is used."""
    available = [
        c["name"] if isinstance(c, dict) else c.name for c in class_info or []
    ]
    for name, sig_class, sig_dataset, bkg_names in sig_over_bkg:
        what = f"{name} / ({name} + {' + '.join(bkg_names)})"
        sig_idx = _find_class_idx(class_info, sig_class)
        bkg_idxs = [_find_class_idx(class_info, b) for b in bkg_names]
        if sig_idx is None or None in bkg_idxs:
            print(f"Skipping {what} plot: class not found among {available}")
            continue
        if max([sig_idx] + bkg_idxs) >= n_score:
            print(
                f"Skipping {what} plot: class index out of range of the "
                f"{n_score} output nodes"
            )
            continue
        sig_dataset_idx = None
        if sig_dataset:
            if dataset_idx is None:
                print(f"Skipping {what} plot: no per-event dataset index")
                continue
            sig_dataset_idx = _find_dataset_idx(class_info, sig_idx, sig_dataset)
            if sig_dataset_idx is None:
                print(
                    f"Skipping {what} plot: dataset {sig_dataset} not found "
                    f"in class {sig_class}"
                )
                continue
        yield name, bkg_names, sig_idx, bkg_idxs, sig_dataset_idx


def _sig_over_bkg_masks(labels, dataset_idx, sig_idx, bkg_idxs, sig_dataset_idx):
    """Return the ``(signal, background)`` event masks of a sig_over_bkg entry:
    the events of the signal class (only those of ``sig_dataset_idx`` if not
    ``None``) and those of the background classes."""
    sig_mask = labels == sig_idx
    if sig_dataset_idx is not None:
        sig_mask &= np.asarray(dataset_idx).astype(int) == sig_dataset_idx
    return sig_mask, np.isin(labels, bkg_idxs)


def plot_per_dataset_roc_curves(
    score_lbl_tensor_test,
    dataset_idx_test,
    dir,
    show,
    class_info,
    comet_logger=None,
    kl_bkg_str=None,
    lumitext=LUMITEXT,
):
    """For each class made of several coupling points (e.g. the VBF or ggF HH
    samples), plot the ROC of each of its datasets vs all the events of the
    other classes, using the score of the node of the class. The events of
    the other datasets of the same class are not used.

    Produces ``roc_curve_<class>_datasets_vs_rest`` in ``dir``.
    """
    if class_info is None or not all(
        isinstance(c, dict) and "datasets" in c for c in class_info
    ):
        print("No dataset names in the class info: skipping the per-dataset ROCs.")
        return

    n_score, label_col, weight_col, kl_col = get_layout(score_lbl_tensor_test)
    labels = score_lbl_tensor_test[:, label_col].astype(int)
    abs_weights = abs(score_lbl_tensor_test[:, weight_col])
    kl_values = score_lbl_tensor_test[:, kl_col]
    dataset_idx_test = np.asarray(dataset_idx_test).astype(int)
    annotation = rf"bkg $\kappa_\lambda$ = {kl_bkg_str}" if kl_bkg_str else None
    roc_info_dict = {}

    for c in class_info:
        c_idx = int(c["class_idx"])
        class_mask = labels == c_idx
        # split only the classes made of several coupling points (the kl
        # column is 9999 for the datasets without a coupling, e.g. data or ttbar)
        if (
            c_idx >= n_score
            or len(c["datasets"]) < 2
            or class_mask.sum() == 0
            or np.all(kl_values[class_mask] == 9999.0)
        ):
            continue

        series = {}
        for d_idx, dataset in enumerate(c["datasets"]):
            pos_mask = class_mask & (dataset_idx_test == d_idx)
            mask = pos_mask | ~class_mask
            sub_labels = pos_mask[mask].astype(int)
            if sub_labels.sum() == 0 or (1 - sub_labels).sum() == 0:
                print(f"No test events for dataset {dataset}: skipping its ROC.")
                continue
            scores = score_lbl_tensor_test[mask, c_idx]
            fpr, tpr, _ = roc_curve(sub_labels, scores, sample_weight=abs_weights[mask])
            roc_auc = roc_auc_score(
                sub_labels, scores, sample_weight=abs_weights[mask]
            )

            short_name = _short_dataset_name(dataset)
            print(f"{c['name']} {short_name} vs rest: AUC = {roc_auc:.3f}")
            series[f"{short_name} (AUC = {roc_auc:.3f})"] = {
                "data": {"x": [tpr, None], "y": [fpr, None]},
                "style": {"linestyle": "-", "markersize": 0},
            }
            roc_info_dict[f"tpr_{c['name']}_{short_name}"] = tpr
            roc_info_dict[f"fpr_{c['name']}_{short_name}"] = fpr
            roc_info_dict[f"auc_{c['name']}_{short_name}"] = roc_auc

        _save_roc_plot(
            series,
            dir,
            f"roc_curve_{c['name']}_datasets_vs_rest",
            f"roc_curve_{c['name']}_datasets_vs_rest",
            show,
            comet_logger=comet_logger,
            lumitext=lumitext,
            annotation=f"{c['name']} vs rest"
            + (f"\n{annotation}" if annotation else ""),
        )

    np.savez(f"{dir}/tpr_fpr_per_dataset.npz", **roc_info_dict)


def plot_multiclass_sig_over_bkg_roc_curves(
    score_lbl_tensor_test,
    dir,
    show,
    sig_over_bkg,
    class_info=None,
    comet_logger=None,
    kl_bkg_str=None,
    lumitext=LUMITEXT,
    dataset_idx_test=None,
):
    """For every entry of ``sig_over_bkg`` plot the ROC of the signal vs the
    sum of the backgrounds, using the discriminant
    D = p(signal) / (p(signal) + sum_b p(b)) as the score. Only the test events
    of the signal (dataset) or of one of the backgrounds are used.
    Produces ``roc_sig_over_bkg_<name>`` per entry."""
    n_score, label_col, weight_col, _ = get_layout(score_lbl_tensor_test)
    labels = score_lbl_tensor_test[:, label_col].astype(int)
    abs_weights = abs(score_lbl_tensor_test[:, weight_col])
    annotation = rf"bkg $\kappa_\lambda$ = {kl_bkg_str}" if kl_bkg_str else None
    roc_info_dict = {}

    for sig_name, bkg_names, sig_idx, bkg_idxs, sig_d_idx in _resolve_sig_over_bkg(
        sig_over_bkg, class_info, n_score, dataset_idx_test
    ):
        sig_mask, bkg_mask = _sig_over_bkg_masks(
            labels, dataset_idx_test, sig_idx, bkg_idxs, sig_d_idx
        )
        mask = sig_mask | bkg_mask
        roc_labels = sig_mask[mask].astype(int)
        if roc_labels.sum() == 0 or (1 - roc_labels).sum() == 0:
            print(f"Skipping {sig_name} ROC: no signal or background test events")
            continue
        d_test = _sig_over_bkg_discriminant(
            score_lbl_tensor_test[mask], sig_idx, bkg_idxs
        )
        fpr, tpr, _ = roc_curve(roc_labels, d_test, sample_weight=abs_weights[mask])
        roc_auc = roc_auc_score(roc_labels, d_test, sample_weight=abs_weights[mask])

        sig_label = sig_name
        bkg_label = " + ".join(_class_label(class_info, b) for b in bkg_idxs)
        print(f"{sig_label} vs {bkg_label}: AUC = {roc_auc:.3f}")

        tag = _normalize_class_name(sig_name)
        roc_info_dict[f"tpr_{tag}"] = tpr
        roc_info_dict[f"fpr_{tag}"] = fpr
        roc_info_dict[f"auc_{tag}"] = roc_auc

        _save_roc_plot(
            {
                f"{sig_label} vs {bkg_label} (AUC = {roc_auc:.3f})": {
                    "data": {"x": [tpr, None], "y": [fpr, None]},
                    "style": {"linestyle": "-", "markersize": 0},
                }
            },
            dir,
            f"roc_sig_over_bkg_{tag}",
            f"roc_sig_over_bkg_{tag}",
            show,
            comet_logger=comet_logger,
            lumitext=lumitext,
            annotation=annotation,
        )

    np.savez(f"{dir}/tpr_fpr_sig_over_bkg.npz", **roc_info_dict)


def plot_multiclass_sig_over_bkg(
    score_lbl_tensor_train,
    score_lbl_tensor_test,
    dir,
    show,
    sig_over_bkg,
    class_info=None,
    comet_logger=None,
    kl_bkg_str=None,
    lumitext=LUMITEXT,
    dataset_idx_train=None,
    dataset_idx_test=None,
):
    """For every entry of ``sig_over_bkg`` build the discriminant

        D = p(signal) / (p(signal) + sum_b p(b))

    from the softmax outputs, i.e. a monotonic transformation of
    ``p(signal) / sum_b p(b)`` bounded in [0, 1]. Only the events of the
    signal (dataset) or of one of the backgrounds are used. Produces
    ``sig_over_bkg_<name>{,_log}``: the normalized distribution of D for the
    signal and for each background, training vs test. The corresponding ROCs
    are drawn by :func:`plot_multiclass_sig_over_bkg_roc_curves`.
    """
    n_score, label_col, weight_col, _ = get_layout(score_lbl_tensor_test)

    def normalized_hist(scores, weights):
        h = Hist.new.Reg(30, 0, 1, name="score").Weight()
        total = np.sum(weights)
        h.fill(scores, weight=weights / total if total else weights)
        return h

    train_labels = score_lbl_tensor_train[:, label_col].astype(int)
    test_labels = score_lbl_tensor_test[:, label_col].astype(int)
    train_weights = score_lbl_tensor_train[:, weight_col]
    test_weights = score_lbl_tensor_test[:, weight_col]
    has_dataset_idx = dataset_idx_train is not None and dataset_idx_test is not None

    for sig_name, bkg_names, sig_idx, bkg_idxs, sig_d_idx in _resolve_sig_over_bkg(
        sig_over_bkg,
        class_info,
        n_score,
        dataset_idx_test if has_dataset_idx else None,
    ):
        tag = _normalize_class_name(sig_name)
        xlabel = f"{sig_name} / ({' + '.join([sig_name] + bkg_names)}) score"

        d_train = _sig_over_bkg_discriminant(score_lbl_tensor_train, sig_idx, bkg_idxs)
        d_test = _sig_over_bkg_discriminant(score_lbl_tensor_test, sig_idx, bkg_idxs)
        sig_train, _ = _sig_over_bkg_masks(
            train_labels, dataset_idx_train, sig_idx, bkg_idxs, sig_d_idx
        )
        sig_test, _ = _sig_over_bkg_masks(
            test_labels, dataset_idx_test, sig_idx, bkg_idxs, sig_d_idx
        )
        components = [(sig_name, sig_train, sig_test)] + [
            (_class_label(class_info, b), train_labels == b, test_labels == b)
            for b in bkg_idxs
        ]

        # --- distributions of the discriminant, training vs test ---
        series_dict = {}
        annotations = []
        for j, (label, tr_mask, te_mask) in enumerate(components):
            color = CMS_COLORS[j % len(CMS_COLORS)]
            if tr_mask.sum() == 0 or te_mask.sum() == 0:
                continue

            hist_train = normalized_hist(d_train[tr_mask], train_weights[tr_mask])
            hist_test = normalized_hist(d_test[te_mask], test_weights[te_mask])
            series_dict[f"{label} (training)"] = {
                "data": hist_train,
                "style": {"color": color, "histtype": "step"},
            }
            series_dict[f"{label} (test)"] = {
                "data": hist_test,
                "style": {"color": color, "histtype": "errorbar"},
            }

            _, p_value = stats.ks_2samp(d_train[tr_mask], d_test[te_mask])
            chi2, pvalue_chi2 = chi_square(hist_test, hist_train)
            annotations.append(
                {
                    "s": f"KS p-value = {p_value:.2f}, "
                    + r"$\chi^2$/ndof= {:.1f},".format(chi2)
                    + f"  p-value= {pvalue_chi2:.2f}",
                    "color": color,
                }
            )

        if not series_dict:
            print(f"Skipping {xlabel} plot: no events of these classes")
            continue

        if kl_bkg_str:
            annotations.append({"s": rf"bkg $\kappa_\lambda$ = {kl_bkg_str}"})

        base = f"{dir}/sig_over_bkg_{tag}"
        for log in [False, True]:
            plotter = (
                HEPPlotter("CMS")
                .set_plot_config(
                    figsize=[13, 13], lumitext=lumitext, cmstext="Private"
                )
                .set_output(f"{base}{'_log' if log else ''}")
                .set_labels(xlabel=xlabel, ylabel="Normalized counts")
                .set_data(series_dict, plot_type="1d")
                .set_options(
                    legend_loc="upper left",
                    legend_font_size=18,
                    split_legend=False,
                    grid=True,
                    y_log=log,
                    ylim_bottom_value=1e-4 if log else 0.0,
                    # leave room for the legend and the annotations
                    ylim_top_factor=3 if log else 2.2,
                )
            )
            for i, annotation in enumerate(annotations):
                plotter.add_annotation(
                    x=0.98,
                    y=0.95 - 0.045 * i,
                    fontsize=18,
                    ha="right",
                    va="center",
                    **annotation,
                )
            if show:
                plotter.show()
            plotter.run()

            if comet_logger:
                comet_logger.log_image(
                    f"{base}{'_log' if log else ''}.png",
                    name=f"sig_over_bkg_{tag}{'_log' if log else ''}",
                )


def plot_kl_distributions_with_fallback(
    score_lbl_train, score_lbl_test, out_dir, test_fraction, **plot_kwargs
):
    """plot_kl_distributions for all the background kl values and for
    background kl = 1, or only for all the background kl values if the
    latter fails (e.g. no background events with kl = 1)."""
    try:
        plot_kl_distributions(
            score_lbl_train,
            score_lbl_test,
            out_dir,
            ["all", 1.0],
            test_fraction,
            **plot_kwargs,
        )
    except ValueError as e:
        logging.getLogger(__name__).warning(
            f"Plotting for background kl = 1 failed ({type(e).__name__}: {e}). "
            "Plotting only for all the background kl values."
        )
        # remove the incomplete plots of the failed kl = 1 attempt
        for prefix in ["sig_bkg", "roc", "calibration"]:
            shutil.rmtree(f"{out_dir}/{prefix}_bkgkl_1.00", ignore_errors=True)
        plot_kl_distributions(
            score_lbl_train,
            score_lbl_test,
            out_dir,
            ["all"],
            test_fraction,
            **plot_kwargs,
        )


def main():
    # parse the arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i", "--input-dir", default="score_lbls", help="Input directory", type=str
    )
    parser.add_argument(
        "-s", "--show", default=False, help="Show plots", action="store_true"
    )
    parser.add_argument(
        "-r",
        "--rescale",
        nargs="+",
        type=float,
        default=[
            # 0.3363,
            # 0.3937,  # this is the ratio of the (new xsec * BR) over the (old xsec)
        ],  # 2.889e-6 4.567e-5 (=1/sumgenweights*10) #9.71589e-7, 1.79814e-5] #  3.453609602837785e-05,0.00017658439204048897,
        help="Rescale the signal and background when computing the number of expected events",
    )
    parser.add_argument(
        "-e", "--signal-eff", default=-1, help="Signal efficiency to cut", type=float
    )
    parser.add_argument(
        "-klb",
        "--kl-background",
        nargs="+",
        default=["all", "1"],
        help="Background kl values to plot. Use 'all' for the inclusive plot, numbers for specific kl values, or 'full' to plot every available kl (default: all 1).",
    )
    parser.add_argument(
        "--sig-over-bkg",
        nargs="+",
        default=[
            f"{name}={sig_class}{':' + dataset if dataset else ''}={'+'.join(bkgs)}"
            for name, sig_class, dataset, bkgs in SIG_OVER_BKG
        ],
        help="Multi-class only: signal / (signal + sum of backgrounds) "
        "discriminants to plot, given as NAME=CLASS[:DATASET]=BKG1+BKG2 with the "
        "class names of the config and the short name of a dataset of the "
        "signal class (default: %(default)s).",
    )

    parser.print_help()
    args = parser.parse_args()

    input_file = f"{args.input_dir}/score_lbl_array.npz"

    # load the labels and scores from the train and test datasets from a .npz file
    score_lbl_tensor_train = np.load(input_file, allow_pickle=True)[
        "score_lbl_array_train"
    ]
    score_lbl_tensor_test = np.load(input_file, allow_pickle=True)[
        "score_lbl_array_test"
    ]

    try:
        train_test_fractions = np.load(input_file, allow_pickle=True)[
            "train_test_fractions"
        ]
    except KeyError:
        train_test_fractions = [0.8, 0.1]

    try:
        class_info = list(np.load(input_file, allow_pickle=True)["class_info"])
    except KeyError:
        class_info = None

    # per-event dataset index, used for the per-dataset ROCs
    try:
        dataset_idx_train = np.load(input_file, allow_pickle=True)["dataset_idx_train"]
        dataset_idx_test = np.load(input_file, allow_pickle=True)["dataset_idx_test"]
    except KeyError:
        dataset_idx_train = dataset_idx_test = None

    sig_over_bkg = []
    for entry in args.sig_over_bkg:
        name, sig_spec, bkgs = entry.split("=", 2)
        sig_class, _, dataset = sig_spec.partition(":")
        sig_over_bkg.append((name, sig_class, dataset or None, bkgs.split("+")))

    # resolve background kl values to plot
    try:
        _, label_col, _, kl_col = get_layout(score_lbl_tensor_train)
        bkg_mask = score_lbl_tensor_train[:, label_col] == 0
        kl_bkg_unique_values = list(
            np.unique(score_lbl_tensor_train[bkg_mask, kl_col])
        )
    except IndexError:
        kl_bkg_unique_values = [9999.0]

    if "full" in args.kl_background:
        kls_background_to_plot = ["all"] + kl_bkg_unique_values
    else:
        kl_bkg_requested = set()
        for v in args.kl_background:
            kl_bkg_requested.add(v if v == "all" else float(v))
        kls_background_to_plot = [
            kl
            for kl in ["all"] + kl_bkg_unique_values
            if (kl if kl == "all" else float(kl)) in kl_bkg_requested
        ]
    print(f"Background kl values to plot: {kls_background_to_plot}")

    plot_kl_distributions(
        score_lbl_tensor_train,
        score_lbl_tensor_test,
        args.input_dir,
        kls_background_to_plot,
        train_test_fractions[1],
        show=args.show,
        rescale=args.rescale,
        signal_eff=args.signal_eff,
        get_max_significance=False,
        # get the years from the config saved in the training directory
        lumitext=get_lumitext_from_dir(args.input_dir),
        class_info=class_info,
        sig_over_bkg=sig_over_bkg,
        dataset_idx_train=dataset_idx_train,
        dataset_idx_test=dataset_idx_test,
    )


if __name__ == "__main__":
    main()
