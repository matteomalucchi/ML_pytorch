"""Fold-to-fold spread of the per-fold reweighting weights of a k-fold
training (ml_train with k_folds >= 2 and onnx_logdiff: True).

Every per-fold _logdiff.onnx model (the log-weight Delta_k = z_1 - z_0 =
log(p1 / p0) of fold k) is evaluated on the events of the k-fold training
(by default the out-of-fold test events of all the folds, i.e. every event
once, randomly subsampled to --max-events; with --random-events N a random
subset of N events of the k-fold data), so that every event has the
log-weights of all the K folds. With them the script checks whether the
fold-to-fold scatter of the weights is symmetric in w or in log w, and
compares the fold combinations average, median (of w), logaverage and
logmedian (of log w) of ml_kfold_merge, before any clamp.

Outputs in <input>/kfold/weight_diagnostics/ (or --output):
  - delta_spread[_logy].png    : Delta_k - mean_k Delta_k, all folds stacked
  - delta_spread_bins.png      : the same in bins of mean_k Delta_k (quantiles
                                 and the tails above log 10 and log 100), with
                                 the per-bin RMS
  - delta_rms_vs_mean.png      : RMS of Delta_k - mean_k Delta_k vs mean_k
                                 Delta_k
  - w_rel_spread[_logy].png    : (w_k - mean_k w_k) / mean_k w_k, for the
                                 comparison with the spread in log w
  - weights_combinations.png   : weights of average, median, logaverage and
                                 logmedian (log x), with mean, std and max
  - average_vs_logaverage.png  : 2D histogram of the average vs the
                                 logaverage weight
  - weight_tails.txt / .yml    : fraction of events with w > 10, 50, 100,
                                 1000 for every combination, the fraction
                                 clamped by weight_log_clamp and the
                                 skewness of the spread in log w and in w
"""

import argparse
import logging
import os
import shutil

import numpy as np
from omegaconf import OmegaConf

from ml_pytorch.scripts.kfold_merge import (
    find_best_onnx,
    fold_log_weights,
    get_fold_dirs,
    get_kfold_data,
)

logger = logging.getLogger(__name__)

THRESHOLDS = [10, 50, 100, 1000]
COMBINATIONS = ["average", "median", "logaverage", "logmedian"]


def load_cfg(main_dir, config=None):
    """Default configs merged with the training config (as ml_kfold_merge)."""
    file_dir = os.path.dirname(__file__)
    cfg = OmegaConf.load(f"{file_dir}/../defaults/default_configs.yml")
    if config is None:
        config = next(
            f
            for f in [
                f"{main_dir}/config_parameters.yml",
                f"{main_dir}/fold_0/config_parameters.yml",
            ]
            if os.path.exists(f)
        )
    return OmegaConf.merge(cfg, OmegaConf.load(config))


def select_rows(fold_id, labels, max_events, random_events, label, seed):
    """Rows (sorted) of the events to evaluate: the out-of-fold test events
    of all the folds (every event, fold_id[row] is the fold it is a test
    event of) subsampled to max_events or, with random_events, a random
    subset of random_events events; only the class ``label`` if not None."""
    rng = np.random.default_rng(seed)
    candidates = np.arange(len(fold_id))
    if label is not None:
        candidates = candidates[labels == label]
    n = random_events if random_events else max_events
    if n and n < len(candidates):
        candidates = rng.choice(candidates, n, replace=False)
    return np.sort(candidates)


def combinations(delta, fold_id):
    """log of the weight of every fold combination (before any clamp)."""
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        w = np.exp(delta)
        out = {
            "average": np.log(np.mean(w, axis=0)),
            "median": np.log(np.median(w, axis=0)),
            "logaverage": np.mean(delta, axis=0),
            "logmedian": np.median(delta, axis=0),
        }
    # weight of the fold the event is a test event of (the event model with
    # kfold_split event)
    out["oof_fold"] = np.choose(fold_id, delta)
    return out


def skewness(x):
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return float("nan")
    s = np.std(x)
    return float(np.mean((x - np.mean(x)) ** 3) / s**3) if s > 0 else 0.0


def rms(x):
    return float(np.sqrt(np.mean(x**2))) if len(x) else float("nan")


def mean_delta_bins(mean_delta):
    """Bin edges of mean_k Delta_k: quartiles of the events below log 10,
    then [log 10, log 100] and above log 100."""
    log10, log100 = np.log(10.0), np.log(100.0)
    core = mean_delta[mean_delta < log10]
    q = np.quantile(core, [0.25, 0.5, 0.75]) if len(core) else []
    edges = [-np.inf] + [v for v in q if v < log10] + [log10, log100, np.inf]
    return np.unique(edges)


def bin_label(lo, hi):
    def fmt(v):
        if v == np.log(10.0):
            return "log 10"
        if v == np.log(100.0):
            return "log 100"
        return f"{v:.2f}"

    if not np.isfinite(lo):
        return rf"$\bar\Delta$ < {fmt(hi)}"
    if not np.isfinite(hi):
        return rf"$\bar\Delta$ > {fmt(lo)}"
    return rf"{fmt(lo)} < $\bar\Delta$ < {fmt(hi)}"


def make_plots(delta, combos, out_dir, weight_log_clamp=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k_folds = delta.shape[0]
    mean_delta = np.mean(delta, axis=0)
    dev = delta - mean_delta
    dev_flat = dev.ravel()
    finite = np.isfinite(dev_flat)

    # spread in log w
    lim = np.quantile(np.abs(dev_flat[finite]), 0.9999) if finite.any() else 1.0
    bins = np.linspace(-lim, lim, 101)
    for logy in [False, True]:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(np.clip(dev_flat[finite], -lim, lim), bins=bins, histtype="step", lw=1.5,
                label=f"RMS {rms(dev_flat[finite]):.3f}, skewness {skewness(dev_flat):.3f}")
        ax.set_xlabel(r"$\Delta_k - \bar\Delta$  ($\Delta_k = \log w_k$)")
        ax.set_ylabel(f"Events x folds ({k_folds} folds)")
        if logy:
            ax.set_yscale("log")
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"{out_dir}/delta_spread{'_logy' if logy else ''}.png", dpi=150)
        plt.close(fig)

    # spread in w, relative to the average weight
    with np.errstate(over="ignore", invalid="ignore"):
        w = np.exp(delta)
        w_mean = np.mean(w, axis=0)
        rel = ((w - w_mean) / w_mean).ravel()
    rel = rel[np.isfinite(rel)]
    rel_lim = np.quantile(np.abs(rel), 0.9999) if len(rel) else 1.0
    rel_bins = np.linspace(-rel_lim, rel_lim, 101)
    for logy in [False, True]:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(np.clip(rel, -rel_lim, rel_lim), bins=rel_bins, histtype="step", lw=1.5,
                label=f"RMS {rms(rel):.3f}, skewness {skewness(rel):.3f}")
        ax.set_xlabel(r"$(w_k - \bar w) / \bar w$")
        ax.set_ylabel(f"Events x folds ({k_folds} folds)")
        if logy:
            ax.set_yscale("log")
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"{out_dir}/w_rel_spread{'_logy' if logy else ''}.png", dpi=150)
        plt.close(fig)

    # spread in bins of mean Delta
    edges = mean_delta_bins(mean_delta[np.isfinite(mean_delta)])
    bin_stats = []
    fig, ax = plt.subplots(figsize=(8, 6))
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (mean_delta >= lo) & (mean_delta < hi)
        d = dev[:, sel].ravel()
        d = d[np.isfinite(d)]
        stats = {
            "low": float(lo),
            "high": float(hi),
            "n_events": int(sel.sum()),
            "rms": rms(d),
            "skewness": skewness(d),
        }
        bin_stats.append(stats)
        if len(d) == 0:
            continue
        ax.hist(np.clip(d, -lim, lim), bins=bins, histtype="step", lw=1.5, density=True,
                label=f"{bin_label(lo, hi)}: {sel.sum()} ev., RMS {stats['rms']:.3f}")
    ax.set_xlabel(r"$\Delta_k - \bar\Delta$")
    ax.set_ylabel("Normalised")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/delta_spread_bins.png", dpi=150)
    plt.close(fig)

    # RMS of the spread vs mean Delta (fine quantile bins)
    fin = np.isfinite(mean_delta)
    fine_edges = np.unique(np.quantile(mean_delta[fin], np.linspace(0, 1, 41)))
    centers, rms_values, n_values = [], [], []
    for lo, hi in zip(fine_edges[:-1], fine_edges[1:]):
        sel = fin & (mean_delta >= lo) & (mean_delta <= hi)
        d = dev[:, sel].ravel()
        d = d[np.isfinite(d)]
        if len(d):
            centers.append(np.median(mean_delta[sel]))
            rms_values.append(rms(d))
            n_values.append(int(sel.sum()))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(centers, rms_values, "o-")
    for v, name in [(np.log(10.0), "log 10"), (np.log(100.0), "log 100")]:
        ax.axvline(v, color="gray", ls="--", lw=1)
        ax.text(v, ax.get_ylim()[1], name, rotation=90, va="top", ha="right", fontsize=8)
    ax.set_xlabel(r"$\bar\Delta = \frac{1}{K}\sum_k \log w_k$")
    ax.set_ylabel(r"RMS of $\Delta_k - \bar\Delta$")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/delta_rms_vs_mean.png", dpi=150)
    plt.close(fig)

    # weights of the combinations
    fig, ax = plt.subplots(figsize=(8, 6))
    all_logw = np.concatenate([combos[c][np.isfinite(combos[c])] for c in COMBINATIONS])
    lo, hi = np.quantile(all_logw, [1e-5, 1.0]) if len(all_logw) else (-1, 1)
    wbins = np.logspace(lo / np.log(10) - 0.1, hi / np.log(10) + 0.1, 101)
    for c in COMBINATIONS:
        with np.errstate(over="ignore"):
            wc = np.exp(combos[c])
        wc = wc[np.isfinite(wc)]
        ax.hist(wc, bins=wbins, histtype="step", lw=1.5,
                label=f"{c}: mean {np.mean(wc):.3g}, std {np.std(wc):.3g}, max {np.max(wc):.3g}")
    if weight_log_clamp is not None:
        for v in [np.exp(-weight_log_clamp), np.exp(weight_log_clamp)]:
            ax.axvline(v, color="gray", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Weight (before the clamp)")
    ax.set_ylabel("Events")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/weights_combinations.png", dpi=150)
    plt.close(fig)

    # average vs logaverage
    x = combos["average"] / np.log(10)
    y = combos["logaverage"] / np.log(10)
    ok = np.isfinite(x) & np.isfinite(y)
    fig, ax = plt.subplots(figsize=(7, 6))
    if ok.any():
        lo = min(x[ok].min(), y[ok].min())
        hi = max(x[ok].max(), y[ok].max())
        h = ax.hist2d(x[ok], y[ok], bins=100, range=[[lo, hi], [lo, hi]],
                      norm=matplotlib.colors.LogNorm(), cmap="viridis")
        fig.colorbar(h[3], ax=ax, label="Events")
        ax.plot([lo, hi], [lo, hi], "r--", lw=1, label="y = x")
        ax.legend()
    ax.set_xlabel(r"$\log_{10}$ w (average)")
    ax.set_ylabel(r"$\log_{10}$ w (logaverage)")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/average_vs_logaverage.png", dpi=150)
    plt.close(fig)

    return {
        "spread_log_w": {"rms": rms(dev_flat[finite]), "skewness": skewness(dev_flat)},
        "spread_rel_w": {"rms": rms(rel), "skewness": skewness(rel)},
        "spread_log_w_bins": bin_stats,
        "rms_vs_mean_delta": {
            "mean_delta": [float(v) for v in centers],
            "rms": rms_values,
            "n_events": n_values,
        },
    }


def tail_table(combos, weight_log_clamp):
    """Fraction of events with w > THRESHOLDS and clamped by weight_log_clamp
    for every combination."""
    table = {}
    for c, logw in combos.items():
        row = {f"w>{t}": float(np.mean(logw > np.log(t))) for t in THRESHOLDS}
        if weight_log_clamp is not None:
            row["clamped"] = float(np.mean(np.abs(logw) > weight_log_clamp))
        row["max_w"] = float(np.exp(np.nanmax(logw))) if len(logw) else float("nan")
        table[c] = row
    return table


def format_table(table, weight_log_clamp, n_events):
    cols = [f"w>{t}" for t in THRESHOLDS] + (
        ["clamped"] if weight_log_clamp is not None else []
    ) + ["max_w"]
    lines = [
        f"{n_events} events; weights before the clamp; "
        + (
            f"clamped = |log w| > weight_log_clamp = {weight_log_clamp} "
            f"(w outside [{np.exp(-weight_log_clamp):.4g}, {np.exp(weight_log_clamp):.4g}])"
            if weight_log_clamp is not None
            else "no weight_log_clamp configured"
        ),
        f"{'combination':<12}" + "".join(f"{c:>12}" for c in cols),
    ]
    for name, row in table.items():
        lines.append(f"{name:<12}" + "".join(f"{row[c]:>12.4g}" for c in cols))
    return "\n".join(lines)


def run(main_dir, cfg, out_dir, max_events, random_events, label, weight_log_clamp, seed):
    k_folds = cfg.k_folds
    if not k_folds or k_folds < 2:
        raise ValueError(f"k_folds must be >= 2, got {k_folds}")
    fold_dirs = get_fold_dirs(main_dir, k_folds)
    # fail before loading the events if the logdiff models are missing
    files = [find_best_onnx(d, logdiff=True) for d in fold_dirs]
    os.makedirs(out_dir, exist_ok=True)

    cfg, arrays, meta, temp_dir = get_kfold_data(cfg)
    try:
        fold_id = np.asarray(arrays["k"][:, 2]).astype(np.int64)
        labels = np.asarray(arrays["lbl"]).reshape(len(fold_id), -1)[:, 0]
        rows = select_rows(fold_id, labels, max_events, random_events, label, seed)
        logger.info(
            f"Evaluating the {k_folds} logdiff models on {len(rows)} events "
            f"({'random subset' if random_events else 'out-of-fold test events'}"
            f"{'' if label is None else f', class {label}'})"
        )
        delta = np.array(
            [fold_log_weights(f, arrays, rows, logdiff=True) for f in files]
        )
        fold_id = fold_id[rows]
    finally:
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)

    combos = combinations(delta, fold_id)
    stats = make_plots(delta, combos, out_dir, weight_log_clamp)
    table = tail_table(combos, weight_log_clamp)
    text = format_table(table, weight_log_clamp, len(rows))
    text += (
        f"\n\nSpread over the folds: in log w RMS {stats['spread_log_w']['rms']:.4g}, "
        f"skewness {stats['spread_log_w']['skewness']:.4g}; in w (relative) RMS "
        f"{stats['spread_rel_w']['rms']:.4g}, skewness {stats['spread_rel_w']['skewness']:.4g}"
    )
    print(text)
    with open(f"{out_dir}/weight_tails.txt", "w") as f:
        f.write(text + "\n")
    OmegaConf.save(
        OmegaConf.create(
            {
                "n_events": int(len(rows)),
                "k_folds": int(k_folds),
                "events": "random" if random_events else "oof",
                "label": label,
                "weight_log_clamp": weight_log_clamp,
                "tails": table,
                **stats,
            }
        ),
        f"{out_dir}/weight_tails.yml",
    )
    logger.info(f"Weight diagnostics saved in {out_dir}")
    return table, stats


def main():
    parser = argparse.ArgumentParser(
        description="Fold-to-fold spread of the per-fold weights of a k-fold reweighting training (needs onnx_logdiff: True)"
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="Output directory of the k-fold training (containing fold_<k>)",
    )
    parser.add_argument(
        "-c", "--config", default=None,
        help="Training config (default: <input>/config_parameters.yml, else <input>/fold_0/config_parameters.yml)",
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output directory (default: <input>/kfold/weight_diagnostics)",
    )
    parser.add_argument(
        "--max-events", default=2_000_000, type=int,
        help="Out-of-fold test events (all the events of the k-fold data, each a test event of one fold) randomly subsampled to this number (default 2000000; 0 = all)",
    )
    parser.add_argument(
        "--random-events", default=None, type=int,
        help="Use instead a random subset of this number of events of the k-fold data",
    )
    parser.add_argument(
        "--label", default=None, type=int,
        help="Use only the events of this class label (e.g. 0 for the background region, the events that are reweighted; default: all)",
    )
    parser.add_argument(
        "--clamp", default=None, type=float,
        help="weight_log_clamp for the clamped fraction (default: the one of the config)",
    )
    parser.add_argument(
        "--kfold-cache", default=None,
        help="K-fold cache with the events (default: the one of the training if it still exists, otherwise the events are loaded again into a temporary directory)",
    )
    parser.add_argument("--seed", default=0, type=int, help="Seed of the subsampling")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  - %(levelname)s - %(message)s")

    main_dir = args.input.rstrip("/")
    cfg = load_cfg(main_dir, args.config)
    if args.kfold_cache is not None:
        cfg.kfold_cache = args.kfold_cache
    clamp = args.clamp if args.clamp is not None else cfg.get("weight_log_clamp", None)
    run(
        main_dir,
        cfg,
        args.output or f"{main_dir}/kfold/weight_diagnostics",
        args.max_events,
        args.random_events,
        args.label,
        None if clamp is None else float(clamp),
        args.seed,
    )


if __name__ == "__main__":
    main()
