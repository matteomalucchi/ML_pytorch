"""Plot the input variable distributions of signal and background.

The histograms are normalized to unit area and are drawn in the CMS style
(as done in `sig_bkg_eval.py`), together with a signal/background ratio panel.

The plots are produced *before* the training starts and are saved in a
subdirectory of the output directory (`input_variables` by default).

The module can be used either from the training script
(`plot_input_variables_from_loaders`) or as a standalone command line tool
(`ml_input_vars`).
"""

import argparse
import importlib
import logging
import os

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

hep.style.use("CMS")
hep.cms.label(loc=0)

logger = logging.getLogger(__name__)

DEFAULT_SUBDIR = "input_variables"
DEFAULT_BINS = 30
# a variable with less unique values than this is considered discrete
DISCRETE_THRESHOLD = 15
# percentiles used to define the plotting range (robust against outliers)
RANGE_PERCENTILES = (0.1, 99.9)

SIG_COLOR = "blue"
SIG_FACECOLOR = "dodgerblue"
BKG_COLOR = "r"


def to_numpy(array):
    """Convert a torch tensor (or anything array-like) to a numpy array."""
    if hasattr(array, "detach"):
        array = array.detach().cpu()
    if hasattr(array, "numpy"):
        array = array.numpy()
    return np.asarray(array)


def dataset_to_arrays(dataset):
    """Extract features, labels and event weights from a torch dataset.

    The last column of the feature tensor is the event weight
    (see `ml_pytorch.utils.dataset.load_data`).
    """
    features = to_numpy(dataset[:][0])
    labels = to_numpy(dataset[:][1]).flatten()
    weights = features[:, -1]
    features = features[:, :-1]
    return features, labels, weights


def arrays_from_loaders(loaders):
    """Concatenate the datasets of a list of dataloaders (`None` is skipped)."""
    features_list = []
    labels_list = []
    weights_list = []
    for loader in loaders:
        if loader is None:
            continue
        features, labels, weights = dataset_to_arrays(loader.dataset)
        features_list.append(features)
        labels_list.append(labels)
        weights_list.append(weights)

    if not features_list:
        raise ValueError("No dataset available to plot the input variables")

    return (
        np.concatenate(features_list, axis=0),
        np.concatenate(labels_list, axis=0),
        np.concatenate(weights_list, axis=0),
    )


def resolve_variable_names(n_columns, input_variables):
    """Match the name of the input variables to the columns of the feature array.

    A single entry of `input_variables` can correspond to more than one column
    of the feature array (e.g. a jet collection variable which is unflattened
    into one column per jet), in which case the position in the collection is
    appended to the name of the variable.
    """
    n_variables = len(input_variables)

    if n_columns == n_variables:
        return list(input_variables)

    if n_variables > 0 and n_columns % n_variables == 0:
        columns_per_variable = n_columns // n_variables
        logger.warning(
            "Found %d columns for %d input variables: assuming %d columns per variable",
            n_columns,
            n_variables,
            columns_per_variable,
        )
        return [
            f"{variable}_{position}"
            for variable in input_variables
            for position in range(columns_per_variable)
        ]

    logger.warning(
        "Found %d columns for %d input variables: using generic names for the plots",
        n_columns,
        n_variables,
    )
    return [f"feature_{i}" for i in range(n_columns)]


def safe_file_name(variable):
    """Sanitize a variable name so that it can be used as a file name."""
    for char in [":", "/", " ", "(", ")", "[", "]", ","]:
        variable = variable.replace(char, "_")
    return variable


def get_bin_edges(sig_values, bkg_values, bins=DEFAULT_BINS):
    """Compute the bin edges common to signal and background."""
    values = np.concatenate([sig_values, bkg_values])
    values = values[np.isfinite(values)]

    if values.size == 0:
        return np.linspace(0.0, 1.0, bins + 1)

    unique_values = np.unique(values)

    if unique_values.size == 1:
        center = float(unique_values[0])
        half_width = max(abs(center) * 0.05, 0.5)
        return np.linspace(center - half_width, center + half_width, bins + 1)

    if unique_values.size <= min(DISCRETE_THRESHOLD, bins):
        steps = np.diff(unique_values)
        # only if the values are (almost) equally spaced use one bin per value,
        # centered on the value itself, otherwise fall back to a uniform binning
        if np.max(steps) <= 1.5 * np.min(steps):
            step = float(np.min(steps))
            return np.concatenate(
                [unique_values - step / 2, [unique_values[-1] + step / 2]]
            )

    low, high = np.percentile(values, RANGE_PERCENTILES)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = float(np.min(values)), float(np.max(values))
    if high <= low:
        high = low + 1.0

    return np.linspace(low, high, bins + 1)


def weighted_histogram(values, weights, bin_edges, normalize=True):
    """Weighted histogram with statistical errors, normalized to unit area.

    The events outside the histogram range are clipped into the first/last bin
    so that no event is lost in the normalization.
    """
    mask = np.isfinite(values) & np.isfinite(weights)
    values = np.clip(values[mask], bin_edges[0], bin_edges[-1])
    weights = weights[mask]

    counts, _ = np.histogram(values, bins=bin_edges, weights=weights)
    sumw2, _ = np.histogram(values, bins=bin_edges, weights=weights**2)
    errors = np.sqrt(sumw2)

    if normalize:
        integral = np.sum(counts * np.diff(bin_edges))
        if integral != 0:
            counts = counts / integral
            errors = errors / abs(integral)

    return counts, errors


def plot_single_variable(
    variable,
    sig_values,
    bkg_values,
    sig_weights,
    bkg_weights,
    dir,
    bins=DEFAULT_BINS,
    show=False,
    log_scale=False,
    lumitext="2022 (13.6 TeV)",
    sig_label="Signal",
    bkg_label="Background",
    formats=("png", "pdf"),
    comet_logger=None,
):
    """Plot the normalized signal and background distribution of one variable."""
    bin_edges = get_bin_edges(sig_values, bkg_values, bins)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

    sig_counts, sig_errors = weighted_histogram(sig_values, sig_weights, bin_edges)
    bkg_counts, bkg_errors = weighted_histogram(bkg_values, bkg_weights, bin_edges)

    fig, (ax, ax_ratio) = plt.subplots(
        2,
        1,
        figsize=[13, 13],
        sharex=True,
        gridspec_kw={"height_ratios": [2.5, 1]},
    )

    ax.stairs(
        sig_counts,
        bin_edges,
        fill=True,
        facecolor=SIG_FACECOLOR,
        edgecolor=SIG_COLOR,
        alpha=0.5,
        label=sig_label,
    )
    ax.stairs(
        bkg_counts,
        bin_edges,
        fill=False,
        edgecolor=BKG_COLOR,
        hatch="\\\\",
        label=bkg_label,
    )
    ax.errorbar(
        bin_centers,
        sig_counts,
        yerr=sig_errors,
        color=SIG_COLOR,
        linestyle="None",
        marker="None",
    )
    ax.errorbar(
        bin_centers,
        bkg_counts,
        yerr=bkg_errors,
        color=BKG_COLOR,
        linestyle="None",
        marker="None",
    )

    # ratio plot: signal over background
    valid = (bkg_counts != 0) & (sig_counts != 0)
    ratio = np.full_like(sig_counts, np.nan, dtype=float)
    ratio_err = np.full_like(sig_counts, np.nan, dtype=float)
    ratio_band = np.zeros_like(bkg_counts, dtype=float)

    ratio[valid] = sig_counts[valid] / bkg_counts[valid]
    ratio_err[valid] = np.abs(sig_errors[valid] / bkg_counts[valid])
    ratio_band[valid] = np.abs(bkg_errors[valid] / bkg_counts[valid])

    ax_ratio.errorbar(
        bin_centers,
        ratio,
        yerr=ratio_err,
        marker="o",
        color="black",
        linestyle="None",
    )
    ax_ratio.fill_between(
        bin_centers,
        1 - ratio_band,
        1 + ratio_band,
        color=BKG_COLOR,
        alpha=0.2,
        step="mid",
    )
    ax_ratio.axhline(y=1, color="black", linestyle="--")

    finite = np.isfinite(ratio) & np.isfinite(ratio_err)
    if finite.any():
        low = float(np.min(ratio[finite] - ratio_err[finite]))
        high = float(np.max(ratio[finite] + ratio_err[finite]))
        padding = 0.1 * (high - low) if high > low else 0.1
        ax_ratio.set_ylim(max(0.0, low - padding), high + padding)

    max_bin = max(np.max(sig_counts, initial=0), np.max(bkg_counts, initial=0))
    if max_bin > 0:
        ax.set_ylim(bottom=0, top=max_bin * 1.5)

    ax.set_ylabel("Normalized counts")
    ax_ratio.set_xlabel(variable)
    ax_ratio.set_ylabel("Signal/Background")
    ax_ratio.set_xlim(bin_edges[0], bin_edges[-1])

    ax.legend(loc="upper right", fontsize=20, frameon=False)
    ax.grid()
    ax_ratio.grid()

    hep.cms.lumitext(lumitext, ax=ax)
    hep.cms.text(text="Preliminary", ax=ax, loc=0)

    file_name = safe_file_name(variable)
    if comet_logger:
        comet_logger.log_figure(f"input_variables/{variable}", plt)
    for extension in formats:
        fig.savefig(
            f"{dir}/{file_name}.{extension}", bbox_inches="tight", dpi=300
        )

    if log_scale and max_bin > 0:
        positive_counts = np.concatenate([sig_counts, bkg_counts])
        positive_counts = positive_counts[positive_counts > 0]
        bottom = np.min(positive_counts) * 0.1 if positive_counts.size > 0 else 1e-3
        ax.set_yscale("log")
        ax.set_ylim(bottom=bottom, top=max_bin * 100)
        if comet_logger:
            comet_logger.log_figure(f"input_variables/{variable}_log", plt)
        for extension in formats:
            fig.savefig(
                f"{dir}/{file_name}_log.{extension}", bbox_inches="tight", dpi=300
            )

    if show:
        plt.show()
    plt.close(fig)


def plot_input_variables(
    features,
    labels,
    weights,
    input_variables,
    output_dir,
    subdir=DEFAULT_SUBDIR,
    bins=DEFAULT_BINS,
    show=False,
    log_scale=False,
    formats=("png", "pdf"),
    comet_logger=None,
):
    """Plot the normalized distributions of all the input variables.

    Args:
        features: array of shape (n_events, n_variables) with the input variables.
        labels: array of shape (n_events,), 1 for signal and 0 for background.
        weights: array of shape (n_events,) with the event weights.
        input_variables: list with the name of the input variables.
        output_dir: main output directory of the training.
        subdir: subdirectory of `output_dir` where the plots are saved.

    Returns:
        The directory where the plots have been saved.
    """
    features = np.asarray(features)
    labels = np.asarray(labels).flatten()
    weights = np.asarray(weights).flatten()

    if features.shape[0] != labels.shape[0] or features.shape[0] != weights.shape[0]:
        raise ValueError(
            "features, labels and weights must have the same number of events "
            f"({features.shape[0]}, {labels.shape[0]}, {weights.shape[0]})"
        )
    variable_names = resolve_variable_names(features.shape[1], list(input_variables))

    plot_dir = os.path.join(output_dir, subdir) if subdir else output_dir
    os.makedirs(plot_dir, exist_ok=True)

    sig_mask = labels == 1
    bkg_mask = ~sig_mask

    logger.info(
        "Plotting %d input variables for %d signal and %d background events",
        len(variable_names),
        int(np.sum(sig_mask)),
        int(np.sum(bkg_mask)),
    )

    sig_weights = weights[sig_mask]
    bkg_weights = weights[bkg_mask]

    for i, variable in enumerate(variable_names):
        logger.debug("Plotting input variable %s", variable)
        plot_single_variable(
            variable,
            features[sig_mask, i],
            features[bkg_mask, i],
            sig_weights,
            bkg_weights,
            plot_dir,
            bins=bins,
            show=show,
            log_scale=log_scale,
            formats=formats,
            comet_logger=comet_logger,
        )

    logger.info("Input variable distributions saved in %s", plot_dir)

    return plot_dir


def plot_input_variables_from_loaders(
    loaders,
    input_variables,
    output_dir,
    subdir=DEFAULT_SUBDIR,
    bins=DEFAULT_BINS,
    show=False,
    log_scale=False,
    formats=("png", "pdf"),
    comet_logger=None,
):
    """Plot the input variables starting from a list of torch dataloaders."""
    features, labels, weights = arrays_from_loaders(loaders)

    return plot_input_variables(
        features,
        labels,
        weights,
        input_variables,
        output_dir,
        subdir=subdir,
        bins=bins,
        show=show,
        log_scale=log_scale,
        formats=formats,
        comet_logger=comet_logger,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Plot the input variable distributions of signal and background"
    )
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="Path to the configuration file",
        type=str,
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="Output directory (the plots are saved in a subdirectory of it)",
        type=str,
    )
    parser.add_argument(
        "-sd",
        "--subdir",
        default=DEFAULT_SUBDIR,
        help="Subdirectory of the output directory where the plots are saved",
        type=str,
    )
    parser.add_argument(
        "-b",
        "--bins",
        default=DEFAULT_BINS,
        help="Number of bins of the histograms",
        type=int,
    )
    parser.add_argument(
        "-s", "--show", default=False, help="Show plots", action="store_true"
    )
    parser.add_argument(
        "--log-scale",
        default=False,
        help="Save also the histograms with a logarithmic y axis",
        action="store_true",
    )
    args = parser.parse_args()

    from omegaconf import OmegaConf

    from ml_pytorch.utils.dataset import load_data
    from ml_pytorch.utils.setup_logger import setup_logger
    from ml_pytorch.utils.tools import create_DNN_columns_list

    file_dir = os.path.dirname(__file__)
    cfg = OmegaConf.load(f"{file_dir}/../defaults/default_configs.yml")
    cfg_file = OmegaConf.load(args.config)
    for key, val in cfg_file.items():
        cfg[key] = val

    if args.output_dir:
        cfg.output_dir = args.output_dir
    if not cfg.output_dir:
        cfg.output_dir = f"./out/{os.path.basename(args.config).replace('.yml', '')}"

    os.makedirs(cfg.output_dir, exist_ok=True)

    setup_logger(f"{cfg.output_dir}/logger_input_variables.log", cfg.verbosity)

    if isinstance(cfg.input_variables, str):
        dnn_input_variables_file = importlib.import_module(
            f"ml_pytorch.defaults.{cfg.input_variables}"
        )
        cfg.input_variables = create_DNN_columns_list(
            cfg.run2, dnn_input_variables_file.dnn_input_variables
        )

    (training_loader, val_loader, test_loader, _, _) = load_data(cfg, cfg.seed)

    plot_input_variables_from_loaders(
        [training_loader, val_loader, test_loader],
        cfg.input_variables,
        cfg.output_dir,
        subdir=args.subdir,
        bins=args.bins,
        show=args.show,
        log_scale=args.log_scale,
    )


if __name__ == "__main__":
    main()
