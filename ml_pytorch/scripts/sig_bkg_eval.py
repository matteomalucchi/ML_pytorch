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
    bin_width,
    sig_score_test,
    bkg_score_test,
    sig_weight_test,
    bkg_weight_test,
    test_fraction,
    rescale,
):

    signal_cumulative_integral = np.cumsum(counts_test_list[0][::-1] * bin_width)
    # find the bin with the signal efficiency closest to target
    bin_index = np.argmin(np.abs(signal_cumulative_integral[::-1] - sig_eff))
    # get the DNN score for the target signal efficiency
    dnn_score_target = bin_centers[bin_index]
    # compute the background rejection at target signal efficiency
    bkg_rejection = np.sum(counts_test_list[1][:bin_index] * bin_width[bin_index])

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
    significance_above_target = np.sqrt(
        2
        * (
            (n_sig_above_target + n_bkg_above_target)
            * np.log(n_sig_above_target / n_bkg_above_target + 1)
            - n_sig_above_target
        )
    )

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
    plot_significance=False,
    get_max_significance=False,
):
    # plot the signal and background distributions
    sig_score_train, bkg_score_train = handle_arrays(score_lbl_tensor_train, 0)
    sig_score_test, bkg_score_test = handle_arrays(score_lbl_tensor_test, 0)

    print("sig_score_train",sig_score_train, sig_score_train.shape)
    print("bkg_score_train",bkg_score_train, bkg_score_train.shape)
    print("sig_score_test",sig_score_test, sig_score_test.shape)
    print("bkg_score_test",bkg_score_test, bkg_score_test.shape)
    
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

    print("sig_weight_train",sig_weight_train, sig_weight_train.shape)
    print("bkg_weight_train",bkg_weight_train, bkg_weight_train.shape)
    print("sig_weight_test",sig_weight_test, sig_weight_test.shape)
    print("bkg_weight_test",bkg_weight_test, bkg_weight_test.shape)
    
    # fig, ax = plt.subplots()
    fig, (ax, ax_ratio) = plt.subplots(
        2,
        1,
        figsize=[13, 13],
        sharex=True,
        gridspec_kw={"height_ratios": [2.5, 1]},
    )
    sig_train = ax.hist(
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
    bkg_train = ax.hist(
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
    ax.set_ylim(top=max_bin * 2)
    i = 0
    legend_test_list = []
    for (
        score_test,
        weight_test,
        score_train,
        weight_train,
        color,
        label,
        rescale_factor,
    ) in zip(
        [sig_score_test, bkg_score_test],
        [sig_weight_test, bkg_weight_test],
        [sig_score_train, bkg_score_train],
        [sig_weight_train, bkg_weight_train],
        ["blue", "r"],
        ["Signal (test)", "Background (test)"],
        rescale if rescale else [1, 1],
    ):

        # counts_test, bins = np.histogram(
        #     score_test,
        #     weights=weight_test,
        #     bins=30,
        #     density=False,
        #     range=(0, 1),
        # )
        # counts_train, bins = np.histogram(
        #     score_train,
        #     weights=weight_train,
        #     bins=30,
        #     density=False,
        #     range=(0, 1),
        # )
        # bin_centers = 0.5 * (bins[:-1] + bins[1:])
        # # Calculate bin widths
        # bin_width = bins[1:] - bins[:-1]

        # # Calculate standard deviation per bin
        # # Calculate error bars by rescaling standard deviation
        # errors_test = np.sqrt(counts_test) / np.sum(counts_test * bin_width)
        # norm_counts_test = counts_test / np.sum(counts_test * bin_width)
        # errors_train = np.sqrt(counts_train) / np.sum(counts_train * bin_width)
        # norm_counts_train = counts_train / np.sum(counts_train * bin_width)

        # ratio = norm_counts_test / norm_counts_train
        # err_num = np.sqrt(counts_test)
        # err_den = np.sqrt(counts_train)
        # ratio_err_test = np.sqrt(
        #     (err_num / counts_train) ** 2
        #     + (counts_test * err_den / counts_train**2) ** 2
        # )
        # ratio_band_train = err_den / counts_train

        bins = np.linspace(0, 1, 31)
        bin_centers = (bins[1:] + bins[:-1]) / 2
        bin_width = bins[1] - bins[0]
        weight_test = weight_test / (np.sum(weight_test) * bin_width)
        weight_train = weight_train / (np.sum(weight_train) * bin_width)

        idx_train = np.digitize(score_train, bins)
        idx_test = np.digitize(score_test, bins)
        h_test = []
        h_train = []
        err_test = []
        err_train = []

        for j in range(1, len(bins)):
            h_test.append(np.sum(weight_test[idx_test == j]))
            h_train.append(np.sum(weight_train[idx_train == j]))
            err_test.append(np.sqrt(np.sum(weight_test[idx_test == j] ** 2)))
            err_train.append(np.sqrt(np.sum(weight_train[idx_train == j] ** 2)))

        h_test = np.array(h_test)
        h_train = np.array(h_train)
        err_test = np.array(err_test)
        err_train = np.array(err_train)
        
        print("h_test",h_test)
        print("h_train",h_train)
        print("err_test",err_test)
        print("err_train",err_train)
            

        ratio = h_test / h_train
        ratio_err_test = np.sqrt(
            (err_test / h_train) ** 2 + (h_test * err_train / h_train**2) ** 2
        )
        ratio_band_train = err_train / h_train

        legend_test_list.append(
            ax.errorbar(
                bin_centers,
                h_test,
                yerr=err_test,
                marker="o",
                color=color,
                label=label,
                linestyle="None",
            )
        )

        # ratio plot
        ax_ratio.errorbar(
            bin_centers,
            ratio,
            yerr=ratio_err_test,
            marker="o",
            color=color,
            label=label,
            linestyle="None",
        )
        ax_ratio.fill_between(
            bin_centers,
            1 - ratio_band_train,
            1 + ratio_band_train,
            color=color,
            alpha=0.2,
        )
        ax_ratio.axhline(y=1, color="black", linestyle="--")

        # remove empty bins
        mask = (h_test != 0 ) & (h_train != 0)
        h_test_nonzero = h_test[mask]
        h_train_nonzero = h_train[mask]
        err_test_nonzero = err_test[mask]
        err_train_nonzero = err_train[mask]
        
        print("h_test_nonzero",h_test_nonzero)
        print("h_train_nonzero",h_train_nonzero)
        print("err_test_nonzero",err_test_nonzero)
        print("err_train_nonzero",err_train_nonzero)
        

        # compute chi squared
        chi_squared = np.sum(
            (
                (h_test_nonzero - h_train_nonzero)
                / np.sqrt(err_test_nonzero**2 + err_train_nonzero**2)
            )
            ** 2
        )
        ndof = len(bin_centers) - 1
        chi2_norm = chi_squared / ndof
        pvalue = 1 - stats.chi2.cdf(chi_squared, ndof)

        ax.text(
            0.6,
            0.75 - 0.05 * i,
            r"$\chi^2$/ndof= {:.1f},".format(chi2_norm) + f"  p-value= {pvalue:.2f}",
            horizontalalignment="left",
            verticalalignment="center",
            transform=ax.transAxes,
            color=color,
            fontsize=20,
        )

        i += 1

    ks_statistic_sig, p_value_sig = stats.ks_2samp(sig_score_train, sig_score_test)
    ks_statistic_bkg, p_value_bkg = stats.ks_2samp(bkg_score_train, bkg_score_test)
    print(f"\nKS: statistic (sig) = {ks_statistic_sig:.30f}")
    print(f"KS: p-value (sig) = {p_value_sig:.30f}")
    print(f"KS: statistic (bkg) = {ks_statistic_bkg:.30f}")
    print(f"KS: p-value (bkg) = {p_value_bkg:.30f}")

    # print the KS test results on the plot
    ax.text(
        0.6,
        0.925,
        f"KS: p-value = {p_value_sig:.2f}",
        fontsize=20,
        transform=ax.transAxes,
        color="blue",
    )
    ax.text(
        0.6,
        0.85,
        f"KS: p-value = {p_value_bkg:.2f}",
        fontsize=20,
        transform=ax.transAxes,
        color="red",
    )

    # Compute significance

    counts_test_list = []
    for score, weight, rescale_factor in zip(
        [sig_score_test, bkg_score_test],
        [sig_weight_test, bkg_weight_test],
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

    n_sig = np.sum(sig_weight_test) / test_fraction * (rescale[0] if rescale else 1)
    n_bkg = np.sum(bkg_weight_test) / test_fraction * (rescale[1] if rescale else 1)
    significance = n_sig / np.sqrt(n_bkg)
    print(f"\nNumber of signal events in the test dataset: {n_sig}")
    print(f"Number of background events in the test dataset: {n_bkg}")
    print(f"Significance: {significance:.2f}\n")

    if plot_significance:
        if get_max_significance:
            max_significance = -1
            for sig_eff_target in np.linspace(0.0, 1.0, 30):
                # compute the significance for each signal efficiency
                # and find the DNN cut that maximizes the significance
                (infos_significance) = compute_significance(
                    sig_eff_target,
                    counts_test_list,
                    bin_centers,
                    bin_width,
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
                bin_width,
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
        handles_legend.append(line_target)
    handles_legend = [
        sig_train[2][0],
        legend_test_list[0],
        bkg_train[2][0],
        legend_test_list[1],
    ]

    ax_ratio.set_xlabel("Output score")
    ax.set_ylabel("Normalized counts")
    ax_ratio.set_ylabel("Test/Train")
    ax_ratio.set_ylim(0.75, 1.25)

    ax.legend(
        loc="upper left",
        # loc="center",
        # bbox_to_anchor=(0.3, 0.9),
        fontsize=20,
        handles=handles_legend,
        frameon=False,
    )
    ax.grid()
    ax_ratio.grid()
    # plt.plot([0.09, 0.88], [8.35, 8.35], color="lightgray", linestyle="-", transform=plt.gca().transAxes)

    hep.cms.lumitext("2022 (13.6 TeV)", ax=ax)
    hep.cms.text(
        text="Preliminary",
        ax=ax,
        loc=0,
    )
    plt.savefig(f"{dir}/sig_bkg_distributions.png", bbox_inches="tight", dpi=300)
    ax.set_ylim(bottom=1e-2, top=max_bin**4)
    ax.set_yscale("log")
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
    plt.yscale("log")
    
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

    # plot the signal and background distributions
    plot_sig_bkg_distributions(
        score_lbl_tensor_train,
        score_lbl_tensor_test,
        args.input_dir,
        args.show,
        args.rescale,
        train_test_fractions[1],
        plot_significance=False,
        get_max_significance=False,
    )

    plot_roc_curve(score_lbl_tensor_test, args.input_dir, args.show)


if __name__ == "__main__":
    main()
