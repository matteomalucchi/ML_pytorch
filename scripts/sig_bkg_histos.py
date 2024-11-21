import matplotlib.pyplot as plt
import argparse
import numpy as np
import mplhep as hep
from scipy import stats
from sklearn.metrics import roc_curve, roc_auc_score

hep.style.use("CMS")
hep.cms.label(loc=0)


def handle_arrays(score_lbl_tensor):
    sig = score_lbl_tensor[score_lbl_tensor[:, 1] == 1]
    bkg = score_lbl_tensor[score_lbl_tensor[:, 1] == 0]

    sig_score = sig[:, 0]
    bkg_score = bkg[:, 0]

    return sig_score, bkg_score


def plot_sig_bkg_distributions(
    score_lbl_tensor_train, score_lbl_tensor_test, dir, show
):
    # plot the signal and background distributions
    sig_score_train, bkg_score_train = handle_arrays(score_lbl_tensor_train)
    sig_score_test, bkg_score_test = handle_arrays(score_lbl_tensor_test)

    fig, ax = plt.subplots()
    sig_train = plt.hist(
        sig_score_train,
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
        bins=30,
        range=(0, 1),
        histtype="step",
        label="Background (training)",
        density=True,
        color="r",
        fill=False,
        hatch="\\\\",
    )


    max_bin= max(max(sig_train[0]), max(bkg_train[0]))
    #set limit on y-axis
    ax.set_ylim(top=max_bin*1.5)

    legend_test_list = []
    for score, color, label in zip(
        [sig_score_test, bkg_score_test],
        ["blue", "r"],
        ["Signal (test)", "Background (test)"],
    ):
        counts, bins, _ = plt.hist(
            score,
            bins=30,
            alpha=0,
            density=True,
            range=(0, 1),
        )
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        # NOTE: are the errors correct?
        # Calculate bin widths
        bin_widths = bins[1:] - bins[:-1]

        # Calculate counts per bin
        counts_per_bin = counts * len(score) * bin_widths

        # Calculate standard deviation per bin
        std_per_bin = np.sqrt(counts_per_bin)

        # Calculate error bars by rescaling standard deviation
        errors = std_per_bin / np.sum(counts_per_bin)

        legend_test_list.append(
            plt.errorbar(
                bin_centers,
                counts,
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

    # compute the AUC of the ROC curve
    roc_auc= roc_auc_score(score_lbl_tensor_test[:, 1], score_lbl_tensor_test[:, 0])
    # print the AUC on the plot
    plt.text(
        0.5,
        0.75,
        f"AUC = {roc_auc:.2f}",
        fontsize=20,
        transform=plt.gca().transAxes
    )

    plt.xlabel("DNN output")
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

    # plot the signal and background distributions
    plot_sig_bkg_distributions(
        score_lbl_tensor_train, score_lbl_tensor_test, args.input_dir, args.show
    )
