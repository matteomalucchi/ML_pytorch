import os
import matplotlib

matplotlib.use("Agg")
import argparse
import numpy as np
from scipy import stats
from sklearn.metrics import roc_curve, roc_auc_score, auc

from hist import Hist
from utils_configs.plot.HEPPlotter import HEPPlotter

LUMITEXT = "2022 (13.6 TeV)"


def handle_arrays(score_lbl_tensor, column=0):
    sig = score_lbl_tensor[score_lbl_tensor[:, 1] == 1]
    bkg = score_lbl_tensor[score_lbl_tensor[:, 1] == 0]

    sig_value = sig[:, column]
    bkg_value = bkg[:, column]

    return sig_value, bkg_value


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
    kl_bkg_str=None,
):
    # plot the signal and background distributions
    sig_score_train, bkg_score_train = handle_arrays(score_lbl_tensor_train, 0)
    sig_score_test, bkg_score_test = handle_arrays(score_lbl_tensor_test, 0)

    print("sig_score_train", sig_score_train, sig_score_train.shape)
    print("bkg_score_train", bkg_score_train, bkg_score_train.shape)
    print("sig_score_test", sig_score_test, sig_score_test.shape)
    print("bkg_score_test", bkg_score_test, bkg_score_test.shape)

    # get weights
    try:
        sig_weight_train, bkg_weight_train = handle_arrays(score_lbl_tensor_train, 2)
        sig_weight_test, bkg_weight_test = handle_arrays(score_lbl_tensor_test, 2)
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
        sig_kl_train, bkg_kl_train = handle_arrays(score_lbl_tensor_train, 3)
        sig_kl_test, bkg_kl_test = handle_arrays(score_lbl_tensor_test, 3)
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
                    "label": "Sig efficiency {:.2f}\nBkg rejection {:.2f}\nDNN score {:.2f}".format(
                        signal_eff,
                        bkg_rejection,
                        dnn_score_target,
                    ),
                }
            )

        # Single overtraining plot with signal and background overlaid
        hist_sig_train = Hist.new.Reg(50, 0, 1, name="score").Weight()
        hist_sig_train.fill(sig_score_train_kl, weight=sig_weight_train_kl)
        hist_sig_test = Hist.new.Reg(50, 0, 1, name="score").Weight()
        hist_sig_test.fill(sig_score_test_kl, weight=sig_weight_test_kl)
        hist_bkg_train = Hist.new.Reg(50, 0, 1, name="score").Weight()
        hist_bkg_train.fill(bkg_score_train, weight=bkg_weight_train)
        hist_bkg_test = Hist.new.Reg(50, 0, 1, name="score").Weight()
        hist_bkg_test.fill(bkg_score_test, weight=bkg_weight_test)

        series_dict = {
            "Signal (training)": {
                "data": hist_sig_train,
                "style": {
                    "color": "blue",
                    "histtype": "fill",
                    "edgecolor": "blue",
                    "facecolor": "dodgerblue",
                    "alpha": 0.5,
                },
            },
            "Signal (test)": {
                "data": hist_sig_test,
                "style": {"histtype": "errorbar", "color": "blue"},
            },
            "Background (training)": {
                "data": hist_bkg_train,
                "style": {"color": "r", "histtype": "step"},
            },
            "Background (test)": {
                "data": hist_bkg_test,
                "style": {"histtype": "errorbar", "color": "r"},
            },
        }

        base = f"{dir}/sig_bkg_distributions_kl_{kl_tag}"

        # the histograms are normalized to unit integral by HEPPlotter, so
        # the same log range can be used for every training
        for log in [False, True]:
            plotter = (
                HEPPlotter("CMS")
                .set_plot_config(figsize=[13, 13], lumitext=LUMITEXT)
                .set_output(f"{base}{'_log' if log else ''}")
                .set_labels(
                    xlabel="Output score",
                    ylabel="Normalized counts",
                )
                .set_data(series_dict, plot_type="1d")
                .set_options(
                    normalize_1d_histo=True,
                    legend_loc="upper left",
                    legend_font_size=20,
                    split_legend=False,
                    grid=True,
                    y_log=log,
                    ylim_bottom_value=1e-4 if log else 0.0,
                    ylim_top_value=1 if log else None,
                    ylim_top_factor=2,
                )
                .add_annotation(
                    x=0.6,
                    y=0.9,
                    s=f"KS sig: p-value = {p_value_sig:.2f}",
                    fontsize=20,
                    color="blue",
                )
                .add_annotation(
                    x=0.6,
                    y=0.85,
                    s=f"KS bkg: p-value = {p_value_bkg:.2f}",
                    fontsize=20,
                    color="r",
                )
                .add_annotation(
                    x=0.6,
                    y=0.7,
                    s=rf"sig $\kappa_\lambda$ = {kl_str}"
                    + "\n"
                    + rf"bkg $\kappa_\lambda$ = {kl_bkg_str if kl_bkg_str else 'all'}",
                    fontsize=18,
                    ha="right",
                    va="bottom",
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
    score_lbl_tensor_test, dir, show, comet_logger=None, kl_bkg_str=None
):
    sig_score_test, bkg_score_test = handle_arrays(score_lbl_tensor_test, 0)
    sig_lbl_test, bkg_lbl_test = handle_arrays(score_lbl_tensor_test, 1)

    # get the weight
    try:
        sig_weight_test, bkg_weight_test = handle_arrays(score_lbl_tensor_test, 2)
    except IndexError:
        print("WARNING: No weight values found in the input file. Using equal weight.")
        sig_weight_test = np.ones_like(sig_score_test)
        bkg_weight_test = np.ones_like(bkg_score_test)

    print("sig_weight_test", sig_weight_test, sig_weight_test.shape)
    print("bkg_weight_test", bkg_weight_test, bkg_weight_test.shape)

    # get the kl values
    try:
        sig_kl_test, bkg_kl_test = handle_arrays(score_lbl_tensor_test, 3)
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
            .set_plot_config(lumitext=LUMITEXT)
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
    comet_logger=None,
):
    if rescale is None:
        rescale = []
    for kl_bkg in kls_background_to_plot:
        kl_bkg_str = "all" if kl_bkg == "all" else f"{kl_bkg:.2f}"

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
                kl_bkg_str=kl_bkg_str,
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

    # resolve background kl values to plot
    try:
        bkg_mask = score_lbl_tensor_train[:, 1] == 0
        kl_bkg_unique_values = list(np.unique(score_lbl_tensor_train[bkg_mask, 3]))
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
    )


if __name__ == "__main__":
    main()
