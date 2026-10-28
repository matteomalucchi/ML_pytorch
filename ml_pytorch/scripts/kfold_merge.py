"""Merge the folds of a k-fold training (ml_train with k_folds >= 2).

Every fold <k> is trained in <main_dir>/fold_<k> and tested on the events of
fold k. This script:
  - concatenates the test scores of all the folds into the out-of-fold scores
    of all the events (<main_dir>/kfold/score_lbl_array_oof.npz) and plots
    them (score histograms and ROC curves)
  - summarises the metrics of the folds with their mean and standard
    deviation (<main_dir>/kfold/kfold_summary.yml)
  - collects the best ONNX model of each fold and builds the combined ONNX
    models (<main_dir>/kfold_onnx/): average of the outputs, average of the
    ratios of the outputs (reweighting) and, with kfold_split event, the model
    that evaluates every event with the model of the fold event % k_folds
"""

import argparse
import glob
import logging
import os
import re
import shutil

import numpy as np
import onnx
import onnxruntime as ort
from omegaconf import OmegaConf

from ml_pytorch.utils.lumitext import get_lumitext
from ml_pytorch.utils.onnx_tools import build_kfold_models, save_onnx_model

logger = logging.getLogger(__name__)

# file names of the combined ONNX models in <main_dir>/kfold_onnx
KFOLD_ONNX_NAMES = {
    "average": "average_model_kfold.onnx",
    "average_ratio": "average_ratio_model_kfold.onnx",
    "event": "event_fold_model_kfold.onnx",
    "event_ratio": "event_fold_ratio_model_kfold.onnx",
}


def get_fold_dirs(main_dir, k_folds):
    fold_dirs = [f"{main_dir}/fold_{k}" for k in range(k_folds)]
    missing = [d for d in fold_dirs if not os.path.isdir(d)]
    if missing:
        raise FileNotFoundError(f"Missing fold directories: {missing}")
    return fold_dirs


def read_fold_metrics(fold_dir):
    """Best epoch, validation and test metrics of a fold from its log file."""
    log_files = glob.glob(f"{fold_dir}/logger_*.log")
    if not log_files:
        raise FileNotFoundError(f"No log file found in {fold_dir}")
    with open(log_files[0], "r") as f:
        text = f.read()

    metrics = {}
    best = re.findall(
        r"Best epoch # (\d+), (?:best val loss|loss val): ([\d.eE+-]+), "
        r"(?:best val accuracy|accuracy val): ([\d.eE+-]+)",
        text,
    )
    if best:
        metrics["best_epoch"] = int(best[-1][0])
        metrics["val_loss"] = float(best[-1][1])
        metrics["val_accuracy"] = float(best[-1][2])
    loss = re.findall(
        r"Eval epoch # \d+,\s+loss train: ([\d.eE+-]+),\s+loss test: ([\d.eE+-]+)",
        text,
    )
    if loss:
        metrics["train_loss"] = float(loss[-1][0])
        metrics["test_loss"] = float(loss[-1][1])
    acc = re.findall(
        r"Eval epoch # \d+,\s+acc train: ([\d.eE+-]+),\s+acc test: ([\d.eE+-]+)",
        text,
    )
    if acc:
        metrics["train_accuracy"] = float(acc[-1][0])
        metrics["test_accuracy"] = float(acc[-1][1])
    return metrics


def summarise_metrics(fold_metrics):
    summary = {"folds": {f"fold_{k}": m for k, m in enumerate(fold_metrics)}}
    keys = sorted(set().union(*fold_metrics))
    for key in keys:
        if key == "best_epoch":
            continue
        values = np.array([m[key] for m in fold_metrics if key in m])
        summary[f"{key}_mean"] = float(np.mean(values))
        summary[f"{key}_std"] = float(np.std(values))
    return summary


def merge_scores(fold_dirs, out_dir):
    """Concatenate the test (out-of-fold) and training scores of the folds."""
    arrays = [
        np.load(f"{d}/score_lbl_array.npz", allow_pickle=True) for d in fold_dirs
    ]
    merged = {
        "score_lbl_array_test": np.concatenate(
            [a["score_lbl_array_test"] for a in arrays]
        ),
        "dataset_idx_test": np.concatenate([a["dataset_idx_test"] for a in arrays]),
        # fold of each out-of-fold score
        "fold_test": np.concatenate(
            [np.full(len(a["score_lbl_array_test"]), k) for k, a in enumerate(arrays)]
        ),
        "score_lbl_array_train": np.concatenate(
            [a["score_lbl_array_train"] for a in arrays]
        ),
        "dataset_idx_train": np.concatenate([a["dataset_idx_train"] for a in arrays]),
        # the out-of-fold scores cover all the events
        "train_test_fractions": np.array(
            [float(np.mean([a["train_test_fractions"][0] for a in arrays])), 1.0]
        ),
        "num_classes": arrays[0]["num_classes"],
        "class_info": arrays[0]["class_info"],
    }
    out_file = f"{out_dir}/score_lbl_array_oof.npz"
    np.savez(out_file, **merged)
    logger.info(
        f"Saved the out-of-fold scores of {len(merged['score_lbl_array_test'])} "
        f"events in {out_file}"
    )
    return merged


def find_best_onnx(fold_dir):
    onnx_files = sorted(glob.glob(f"{fold_dir}/state_dict/model_best_epoch_*.onnx"))
    if not onnx_files:
        raise FileNotFoundError(
            f"No best model ONNX found in {fold_dir}/state_dict (train with onnx: True)"
        )
    if len(onnx_files) > 1:
        logger.warning(f"Several best models in {fold_dir}, using {onnx_files[-1]}")
    return onnx_files[-1]


def run_onnx(model_file, inputs):
    session = ort.InferenceSession(model_file, providers=["CPUExecutionProvider"])
    names = [i.name for i in session.get_inputs()]
    return session.run(None, dict(zip(names, inputs)))[0]


def ratio_of_outputs(out):
    if out.shape[1] == 1:
        return (out / (1 - out)).flatten()
    return out[:, 1] / out[:, 0]


def check_kfold_models(fold_files, kfold_files, n_features, k_folds):
    """Compare the combined models with the per-fold models on random inputs."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(100, n_features)).astype(np.float32)
    # large event numbers, which are not exact in float32
    event = (2**40 + rng.integers(0, 10**6, size=100)).astype(np.int64)
    fold_out = [run_onnx(f, [x]) for f in fold_files]
    # each event with the model of its fold
    selected = np.choose(event % k_folds, [o.T for o in fold_out]).T

    expected = {"average": np.mean(fold_out, axis=0), "event": selected}
    if "average_ratio" in kfold_files or "event_ratio" in kfold_files:
        fold_ratios = [ratio_of_outputs(o) for o in fold_out]
        expected["average_ratio"] = np.mean(fold_ratios, axis=0)
        expected["event_ratio"] = np.choose(event % k_folds, fold_ratios)

    for key, model_file in kfold_files.items():
        inputs = [x, event] if key.startswith("event") else [x]
        out = run_onnx(model_file, inputs)
        # the ratio of a 1-node model keeps the (N, 1) shape of the output
        if out.size != expected[key].size or not np.allclose(
            out.reshape(expected[key].shape), expected[key], rtol=1e-4, atol=1e-6
        ):
            raise RuntimeError(
                f"The combined model {model_file} does not match the fold models"
            )
        logger.info(f"Checked {model_file} against the fold models")


def merge_onnx(fold_dirs, out_dir, ratio, event_split):
    """Copy the best model of each fold and build the combined models."""
    folds_dir = f"{out_dir}/folds"
    os.makedirs(folds_dir, exist_ok=True)
    fold_files = []
    for k, d in enumerate(fold_dirs):
        fold_file = f"{folds_dir}/model_fold{k}.onnx"
        shutil.copy(find_best_onnx(d), fold_file)
        fold_files.append(fold_file)
        logger.info(f"Fold {k}: copied the best model to {fold_file}")

    onnx_models = [onnx.load(f) for f in fold_files]
    graph_input = onnx_models[0].graph.input[0]
    n_features = graph_input.type.tensor_type.shape.dim[1].dim_value
    graph_output = onnx.shape_inference.infer_shapes(onnx_models[0]).graph.output[0]
    n_outputs = graph_output.type.tensor_type.shape.dim[1].dim_value
    if ratio and n_outputs > 2:
        logger.warning(
            f"The ratio of the outputs is defined for 1 or 2 output nodes, the "
            f"model has {n_outputs}: skipping the ratio models (for the "
            f"multiclass morphing use ml_onnx -ar -mc -i {folds_dir})"
        )
        ratio = False

    combined = build_kfold_models(
        onnx_models, n_features, ratio=ratio, event_split=event_split
    )
    kfold_files = {}
    for key, model in combined.items():
        kfold_files[key] = f"{out_dir}/{KFOLD_ONNX_NAMES[key]}"
        save_onnx_model(model, kfold_files[key])
    check_kfold_models(fold_files, kfold_files, n_features, len(fold_dirs))
    return kfold_files


def merge_kfold(main_dir, cfg, ratio=None, plots=True):
    k_folds = cfg.k_folds
    if not k_folds or k_folds < 2:
        raise ValueError(f"k_folds must be >= 2 to merge the folds, got {k_folds}")
    ratio = cfg.onnx_ratio if ratio is None else ratio
    event_split = cfg.kfold_split == "event"

    fold_dirs = get_fold_dirs(main_dir, k_folds)
    kfold_dir = f"{main_dir}/kfold"
    os.makedirs(kfold_dir, exist_ok=True)

    fold_metrics = [read_fold_metrics(d) for d in fold_dirs]
    summary = summarise_metrics(fold_metrics)
    summary_file = f"{kfold_dir}/kfold_summary.yml"
    OmegaConf.save(OmegaConf.create(summary), summary_file)
    logger.info(f"K-fold summary saved in {summary_file}:\n{OmegaConf.to_yaml(summary)}")
    print(OmegaConf.to_yaml(summary))

    merged = merge_scores(fold_dirs, kfold_dir)

    if plots and (cfg.histos or cfg.roc):
        from ml_pytorch.scripts.sig_bkg_eval import (
            plot_kl_distributions_with_fallback,
        )

        plot_kl_distributions_with_fallback(
            merged["score_lbl_array_train"],
            merged["score_lbl_array_test"],
            kfold_dir,
            merged["train_test_fractions"][1],
            do_histos=cfg.histos,
            do_roc=cfg.roc,
            class_info=merged["class_info"].tolist(),
            lumitext=get_lumitext(cfg),
            dataset_idx_train=merged["dataset_idx_train"],
            dataset_idx_test=merged["dataset_idx_test"],
        )

    if cfg.onnx:
        kfold_files = merge_onnx(
            fold_dirs, f"{main_dir}/kfold_onnx", ratio, event_split
        )
        for key, f in kfold_files.items():
            logger.info(f"K-fold ONNX model ({key}): {f}")

    logger.info(f"Merged the {k_folds} folds in {main_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge the folds of a k-fold training (ml_train with k_folds >= 2)"
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Output directory of the k-fold training (containing fold_<k>)",
    )
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        help="Training config (default: <input>/config_parameters.yml, else <input>/fold_0/config_parameters.yml)",
    )
    parser.add_argument(
        "-k", "--k-folds", default=None, type=int, help="Number of folds (overrides config)"
    )
    parser.add_argument(
        "--ratio",
        default=None,
        action="store_true",
        help="Build also the ONNX models of the average of the ratios of the outputs (reweighting)",
    )
    parser.add_argument(
        "--no-plots", action="store_true", help="Do not plot the out-of-fold scores"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  - %(levelname)s - %(message)s"
    )

    main_dir = args.input.rstrip("/")
    file_dir = os.path.dirname(__file__)
    cfg = OmegaConf.load(f"{file_dir}/../defaults/default_configs.yml")
    config = args.config
    if config is None:
        config = next(
            f
            for f in [
                f"{main_dir}/config_parameters.yml",
                f"{main_dir}/fold_0/config_parameters.yml",
            ]
            if os.path.exists(f)
        )
    cfg = OmegaConf.merge(cfg, OmegaConf.load(config))
    if args.k_folds is not None:
        cfg.k_folds = args.k_folds
    merge_kfold(main_dir, cfg, ratio=args.ratio, plots=not args.no_plots)


if __name__ == "__main__":
    main()
