import matplotlib.pyplot as plt
import argparse
import numpy as np
import mplhep as hep
from scipy import stats
from sklearn.metrics import roc_curve, roc_auc_score, auc

hep.style.use("CMS")
hep.cms.label(loc=0)


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


def compute_significance(
    sig_eff,
    counts_test_list,
    bin_centers,
    bin_widths,
    sig_score_test,
    bkg_score_test,
    sig_weight_test,
    bkg_weight_test,
    test_fraction,
    rescale,
):

    signal_cumulative_integral = np.cumsum(counts_test_list[0][::-1] * bin_widths)
    # find the bin with the signal efficiency closest to target
    bin_index = np.argmin(np.abs(signal_cumulative_integral[::-1] - sig_eff))
    # get the DNN score for the target signal efficiency
    dnn_score_target = bin_centers[bin_index]
    # compute the background rejection at target signal efficiency
    bkg_rejection = np.sum(counts_test_list[1][:bin_index] * bin_widths[bin_index])

    # compute number of signal and background events in the test dataset above the target signal efficiency threshold
    n_sig_above_target = (
        np.sum(sig_weight_test[sig_score_test > dnn_score_target])
        / test_fraction
        * (rescale[0] if rescale else 1)
    )
    n_bkg_above_target = (
        np.sum(bkg_weight_test[bkg_score_test > dnn_score_target])
        / test_fraction
        * (rescale[1] if rescale else 1)
    )
    # significance_above_target = n_sig_above_target / np.sqrt(n_bkg_above_target)
    significance_above_target = np.sqrt(2 * ((n_sig_above_target + n_bkg_above_target) * np.log(n_sig_above_target / n_bkg_above_target + 1) - n_sig_above_target))

    return (
        dnn_score_target,
        bkg_rejection,
        n_sig_above_target,
        n_bkg_above_target,
        significance_above_target,
    )


def plot_sig_bkg_distributions(
    score_lbl_tensor_train,
    score_lbl_tensor_test,
    dir,
    show,
    rescale,
    test_fraction,
    get_max_significance=False,
):
    # plot the signal and background distributions
    sig_score_train, bkg_score_train = handle_arrays(score_lbl_tensor_train, 0)
    sig_score_test, bkg_score_test = handle_arrays(score_lbl_tensor_test, 0)

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

    fig, ax = plt.subplots()
    sig_train = plt.hist(
        sig_score_train,
        weights=sig_weight_train,
        bins=30,
        range=(0, 1),
        histtype="step",
        label="Signal (training)",
        density=True,
        edgecolor="blue",
        facecolor="dodgerblue",
        fill=True,
        alpha=0.5,
    )
    bkg_train = plt.hist(
        bkg_score_train,
        weights=bkg_weight_train,
        bins=30,
        range=(0, 1),
        histtype="step",
        label="Background (training)",
        density=True,
        color="r",
        fill=False,
        hatch="\\\\",
    )

    max_bin = max(max(sig_train[0]), max(bkg_train[0]))
    # set limit on y-axis
    ax.set_ylim(top=max_bin * 1.5)

    legend_test_list = []
    for score, weight, color, label, rescale_factor in zip(
        [sig_score_test, bkg_score_test],
        [sig_weight_test, bkg_weight_test],
        ["blue", "r"],
        ["Signal (test)", "Background (test)"],
        rescale if rescale else [1, 1],
    ):
        counts, bins, _ = plt.hist(
            score,
            weights=np.sign(weight),
            bins=30,
            alpha=0,
            density=False,
            range=(0, 1),
        )
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        # Calculate bin widths
        bin_widths = bins[1:] - bins[:-1]

        # Calculate standard deviation per bin
        std_per_bin = np.sqrt(counts)
        # Calculate error bars by rescaling standard deviation
        errors = std_per_bin / np.sum(counts * bin_widths)
        norm_counts = counts / np.sum(counts * bin_widths)

        legend_test_list.append(
            plt.errorbar(
                bin_centers,
                norm_counts,
                yerr=errors,
                marker="o",
                color=color,
                label=label,
                linestyle="None",
            )
        )
    ks_statistic_sig, p_value_sig = stats.ks_2samp(sig_score_train, sig_score_test)
    ks_statistic_bkg, p_value_bkg = stats.ks_2samp(bkg_score_train, bkg_score_test)

    # print the KS test results on the plot
    plt.text(
        0.5,
        0.925,
        f"KS test: p-value (sig) = {p_value_sig:.2f}",
        fontsize=20,
        transform=plt.gca().transAxes,
    )
    plt.text(
        0.5,
        0.85,
        f"KS test: p-value (bkg) = {p_value_bkg:.2f}",
        fontsize=20,
        transform=plt.gca().transAxes,
    )

    counts_test_list = []
    for score, weight, rescale_factor in zip(
        [sig_score_test, bkg_score_test],
        [sig_weight_test, bkg_weight_test],
        rescale if rescale else [1, 1],
    ):
        counts, bins, _ = plt.hist(
            score,
            weights=weight * rescale_factor,
            bins=1000,
            alpha=0,
            density=True,
            range=(0, 1),
        )
        counts_test_list.append(counts)
        bin_widths = bins[1:] - bins[:-1]
        bin_centers = 0.5 * (bins[:-1] + bins[1:])

    n_sig = np.sum(sig_weight_test) / test_fraction * (rescale[0] if rescale else 1)
    n_bkg = np.sum(bkg_weight_test) / test_fraction * (rescale[1] if rescale else 1)
    significance = n_sig / np.sqrt(n_bkg)
    print(f"\nNumber of signal events in the test dataset: {n_sig}")
    print(f"Number of background events in the test dataset: {n_bkg}")
    print(f"Significance: {significance:.2f}\n")

    if get_max_significance:
        max_significance = -1
        for sig_eff_target in np.linspace(0.0, 1.0, 30):
            # compute the significance for each signal efficiency
            # and find the DNN cut that maximizes the significance
            (infos_significance) = compute_significance(
                sig_eff_target,
                counts_test_list,
                bin_centers,
                bin_widths,
                sig_score_test,
                bkg_score_test,
                sig_weight_test,
                bkg_weight_test,
                test_fraction,
                rescale,
            )
            if infos_significance[-1] > max_significance:
                max_significance = infos_significance[-1]
                print("max_significance", max_significance)
                (
                    dnn_score_target,
                    bkg_rejection,
                    n_sig_above_target,
                    n_bkg_above_target,
                    significance_above_target,
                ) = infos_significance
                sig_eff = sig_eff_target
    else:
        sig_eff = 0.8
        (
            dnn_score_target,
            bkg_rejection,
            n_sig_above_target,
            n_bkg_above_target,
            significance_above_target,
        ) = compute_significance(
            sig_eff,
            counts_test_list,
            bin_centers,
            bin_widths,
            sig_score_test,
            bkg_score_test,
            sig_weight_test,
            bkg_weight_test,
            test_fraction,
            rescale,
        )

    print(
        f"\n###########\nNumber of signal events above {sig_eff:.3f} signal efficiency threshold: {n_sig_above_target:.3f}"
    )
    print(
        f"Number of background events above {sig_eff:.3f} signal efficiency threshold: {n_bkg_above_target:.3f}"
    )
    print(
        f"Significance ({dnn_score_target:.3f} DNN cut): {significance_above_target:.3f}"
    )
    # plot the vertical line for the signal efficiency
    line_target = plt.axvline(
        dnn_score_target,
        color="grey",
        linestyle="--",
        label="Sig efficiency {:.2f}\nBkg rejection {:.2f}\nDNN score {:.2f}\nSignificance {:.2f}".format(
            sig_eff,
            bkg_rejection,
            dnn_score_target,
            significance_above_target,
        ),
    )
    plt.xlabel("Output score")
    plt.ylabel("Normalized counts")
    plt.legend(
        loc="upper left",
        # loc="center",
        # bbox_to_anchor=(0.3, 0.9),
        fontsize=20,
        handles=[
            sig_train[2][0],
            legend_test_list[0],
            bkg_train[2][0],
            legend_test_list[1],
            line_target,
        ],
        frameon=False,
    )
    # plt.plot([0.09, 0.88], [8.35, 8.35], color="lightgray", linestyle="-", transform=plt.gca().transAxes)

    hep.cms.lumitext(
        "2022 (13.6 TeV)",
    )
    hep.cms.text(
        text="Simulation Preliminary",
        loc=0,
    )
    plt.savefig(f"{dir}/sig_bkg_distributions.png", bbox_inches="tight", dpi=300)
    ax.set_ylim(bottom=1e-2, top=max_bin**3)
    plt.yscale("log")
    plt.savefig(f"{dir}/sig_bkg_distributions_log.png", bbox_inches="tight", dpi=300)
    if show:
        plt.show()


def plot_roc_curve(score_lbl_tensor_test, dir, show):
    # plot the ROC curve
    fig, ax = plt.subplots()
    try:
        sample_weights = score_lbl_tensor_test[:, 2]
    except IndexError:
        print("WARNING: No weights found in the input file. Using equal weights.")
        sample_weights = np.ones_like(score_lbl_tensor_test[:, 0])

    fpr, tpr, _ = roc_curve(
        score_lbl_tensor_test[:, 1],
        score_lbl_tensor_test[:, 0],
        sample_weight=sample_weights,
    )
    roc_auc = my_roc_auc(
        score_lbl_tensor_test[:, 1],
        score_lbl_tensor_test[:, 0],
        sample_weight=sample_weights,
    )
    plt.plot(tpr, fpr, label="ROC curve (pos+neg weights AUC = %0.3f)" % roc_auc)

    abs_weights_fpr, abs_weights_tpr, _ = roc_curve(
        score_lbl_tensor_test[:, 1],
        score_lbl_tensor_test[:, 0],
        sample_weight=abs(sample_weights),
    )
    abs_weights_roc_auc = roc_auc_score(
        score_lbl_tensor_test[:, 1],
        score_lbl_tensor_test[:, 0],
        sample_weight=abs(sample_weights),
    )
    plt.plot(
        abs_weights_tpr,
        abs_weights_fpr,
        label="ROC curve (abs weights AUC = %0.3f)" % abs_weights_roc_auc,
    )

    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.xlabel("True positive rate")
    plt.ylabel("False positive rate")
    plt.legend(loc="upper left")
    hep.cms.lumitext(
        "2022 (13.6 TeV)",
    )
    hep.cms.text(
        text="Simulation Preliminary",
        loc=0,
    )
    plt.savefig(f"{dir}/roc_curve.png", bbox_inches="tight", dpi=300)
    if show:
        plt.show()


if __name__ == "__main__":
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
            0.3363,
            0.3937,  # this is the ratio of the (new xsec * BR) over the (old xsec)
        ],  # 2.889e-6 4.567e-5 (=1/sumgenweights*10) #9.71589e-7, 1.79814e-5] #  3.453609602837785e-05,0.00017658439204048897,
        help="Rescale the signal and background when computing the number of expected events",
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
    train_test_fractions = np.load(input_file, allow_pickle=True)[
        "train_test_fractions"
    ]

    # plot the signal and background distributions
    plot_sig_bkg_distributions(
        score_lbl_tensor_train,
        score_lbl_tensor_test,
        args.input_dir,
        args.show,
        args.rescale,
        train_test_fractions[1],
        get_max_significance=False,
    )

    plot_roc_curve(score_lbl_tensor_test, args.input_dir, args.show)
