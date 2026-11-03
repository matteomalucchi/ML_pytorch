"""Merge the folds of a k-fold training (ml_train with k_folds >= 2).

Every fold <k> is trained in <main_dir>/fold_<k> and tested on the events of
fold k. This script:
  - evaluates the model of every fold on its test events in memory, giving
    the out-of-fold scores of all the events, and plots them (score histograms,
    ROC and calibration curves); they are saved
    (<main_dir>/kfold/score_lbl_array_oof.npz) only with --save-numpy
  - checks that the test folds are disjoint, cover all the events and are
    those the folds were trained with
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
import json
import re
import shutil
import tempfile

import numpy as np
import onnx
import onnxruntime as ort
from omegaconf import OmegaConf

from ml_pytorch.utils.lumitext import get_lumitext
from ml_pytorch.utils.onnx_tools import (
    build_kfold_models,
    build_multiclass_morphing_models,
    save_onnx_model,
)

logger = logging.getLogger(__name__)

# file names of the combined ONNX models in <main_dir>/kfold_onnx
# file names of the combined ONNX models in <main_dir>/kfold_onnx, by type:
#   classifier                : the scores of the classifier
#   morphing_ratio            : ratio of the outputs p / (1 - p) (1 node) or
#                               p1 / p0 (2 nodes), for the binary reweighting
#   morphing_ttbar_subtracted : 4-node multiclass morphing weight with the
#                               ttbar subtraction (alpha)
# and combination of the folds:
#   average   : average over the folds (input args_0)
#   all_folds : the output of every fold separately, not averaged (outputs
#               <name>_fold_<k>)
#   event_fold: the output of the fold event % k_folds (inputs args_0, event)
KFOLD_ONNX_TYPES = {
    "": "classifier",
    "_ratio": "morphing_ratio",
    "_morphing": "morphing_ttbar_subtracted",
}
KFOLD_ONNX_NAMES = {
    f"{combination}{suffix}": f"kfold_{kind}_{name}.onnx"
    for suffix, kind in KFOLD_ONNX_TYPES.items()
    for combination, name in [
        ("average", "average"),
        ("all_folds", "all_folds"),
        ("event", "event_fold"),
    ]
}


def final_onnx_name(final_key):
    """File name of the copy of the final model, with its type, e.g.
    final_kfold_morphing_ratio.onnx."""
    suffix = next(
        (s for s in ["_ratio", "_morphing"] if final_key.endswith(s)), ""
    )
    return f"final_kfold_{KFOLD_ONNX_TYPES[suffix]}.onnx"


def final_onnx_key(kfold_onnx, ratio, morphing=False):
    """Key in KFOLD_ONNX_NAMES of the final model chosen with kfold_onnx."""
    if kfold_onnx not in ["average", "event"]:
        raise ValueError(f"kfold_onnx must be average or event, got {kfold_onnx}")
    if morphing:
        return f"{kfold_onnx}_morphing"
    return f"{kfold_onnx}_ratio" if ratio else kfold_onnx


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


def check_fold_configs(fold_dirs):
    """Check that the folds are the folds 0 ... K-1 of the same k-fold split
    (same data, fractions, oversampling, seed, k_folds and kfold_split)."""
    from ml_pytorch.utils.dataset import kfold_cache_hash

    hashes = set()
    for k, d in enumerate(fold_dirs):
        cfg_file = f"{d}/config_parameters.yml"
        if not os.path.exists(cfg_file):
            logger.warning(f"No {cfg_file}: cannot check the config of fold {k}")
            continue
        fold_cfg = OmegaConf.load(cfg_file)
        if fold_cfg.get("fold", None) != k:
            raise ValueError(f"{d} is fold {fold_cfg.get('fold', None)}, not fold {k}")
        hashes.add(kfold_cache_hash(fold_cfg))
    if len(hashes) > 1:
        raise ValueError(
            "The folds were trained with different data, fractions, oversampling, "
            "seed, k_folds or kfold_split: they are not the folds of the same split"
        )
    logger.info(f"Checked that the {len(fold_dirs)} folds are the folds of the same split")


def get_kfold_data(cfg):
    """(cfg, arrays, meta, temp_dir) of the events of the k-fold training:
    the memory-mapped k-fold cache if it still exists (e.g. the merge runs in
    the same job as the folds), otherwise the events are loaded again into a
    temporary cache (temp_dir, to be removed by the caller)."""
    from ml_pytorch.utils.dataset import kfold_cache_exists, load_data, open_kfold_cache
    from ml_pytorch.utils.tools import resolve_input_variables

    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    resolve_input_variables(cfg)
    cache_dir = cfg.get("kfold_cache", None)
    temp_dir = None
    if not (cache_dir and kfold_cache_exists(cache_dir)):
        temp_dir = tempfile.mkdtemp(
            prefix="ml_kfold_merge_cache_", dir=os.environ.get("TMPDIR")
        )
        logger.info(
            f"K-fold cache {cache_dir} not found: loading the events again in {temp_dir}"
        )
        prepare_cfg = OmegaConf.create(OmegaConf.to_container(cfg))
        prepare_cfg.kfold_cache = temp_dir
        prepare_cfg.kfold_prepare = True
        prepare_cfg.kfold_split_dir = None
        load_data(prepare_cfg, prepare_cfg.seed)
        cache_dir = temp_dir
    arrays, meta = open_kfold_cache(cache_dir, cfg)
    return cfg, arrays, meta, temp_dir


def evaluate_onnx(onnx_file, arrays, rows, chunk_size=500_000):
    """score_lbl array ([scores..., label, event weight, kl]) and dataset
    indices of the events ``rows`` (sorted), evaluated with ``onnx_file`` in
    chunks, without writing anything to disk."""
    from ml_pytorch.utils.tools import scores_to_probabilities

    session = ort.InferenceSession(onnx_file, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    outputs, weights = [], []
    for start in range(0, len(rows), chunk_size):
        fts = np.asarray(arrays["fts"][rows[start : start + chunk_size]])
        # the last column is the event weight
        outputs.append(session.run(None, {input_name: fts[:, :-1].astype(np.float32)})[0])
        weights.append(fts[:, -1])
    scores = scores_to_probabilities(np.concatenate(outputs)).numpy()
    labels = np.asarray(arrays["lbl"][rows])
    k = np.asarray(arrays["k"][rows])
    score_lbl = np.concatenate(
        [scores, labels[:, None], np.concatenate(weights)[:, None], k[:, :1]], axis=1
    )
    return score_lbl, k[:, 1].astype(np.int64)


def check_fold_info(fold_dir, fold, test_rows, n_events):
    """Check that the fold was trained with the test events expected from the
    split (record written by ml_train in fold_<k>/kfold_fold_info.json)."""
    from ml_pytorch.utils.dataset import kfold_test_info

    info_file = f"{fold_dir}/kfold_fold_info.json"
    if not os.path.exists(info_file):
        logger.warning(f"No {info_file}: cannot check the test events of fold {fold}")
        return
    with open(info_file) as f:
        info = json.load(f)
    expected = kfold_test_info(test_rows, n_events)
    for key, value in expected.items():
        if info.get(key) != value:
            raise ValueError(
                f"Fold {fold} was trained with different test events than the "
                f"split ({key}: {info.get(key)} instead of {value})"
            )


def merge_scores(cfg, fold_dirs, out_dir, save=False):
    """Out-of-fold scores of all the events: every fold model is evaluated in
    memory on its test events (and on a random fraction 1/(K-1) of its
    distinct training events, about one score per event in total, for the
    comparison with the training dataset in the plots). Also check that the
    test folds are disjoint, cover all the events and are those the folds were
    trained with. The scores are saved to out_dir only if ``save``."""
    from ml_pytorch.utils.dataset import kfold_indices

    cfg, arrays, meta, temp_dir = get_kfold_data(cfg)
    try:
        k_folds = cfg.k_folds
        n_events = meta["n_events"]
        labels = np.asarray(arrays["lbl"])
        fold_id = np.asarray(arrays["k"][:, 2]).astype(np.int64)
        if fold_id.min() < 0 or fold_id.max() >= k_folds:
            raise ValueError(f"Fold ids outside 0 ... {k_folds - 1}")
        counts = np.bincount(fold_id, minlength=k_folds)

        rng = np.random.default_rng(0)
        test, train = [], []
        for k, fold_dir in enumerate(fold_dirs):
            fold_cfg = OmegaConf.create(OmegaConf.to_container(cfg))
            fold_cfg.fold = k
            train_idx, _, test_idx = kfold_indices(
                fold_id, labels, fold_cfg, cfg.seed, meta["is_binary"]
            )
            if len(test_idx) != counts[k] or (fold_id[test_idx] != k).any():
                raise ValueError(f"The test events of fold {k} are not the events of fold {k}")
            check_fold_info(fold_dir, k, test_idx, n_events)

            onnx_file = find_best_onnx(fold_dir)
            test.append(evaluate_onnx(onnx_file, arrays, np.sort(test_idx)))
            train_unique = np.unique(train_idx)
            n_keep = int(np.ceil(len(train_unique) / max(1, k_folds - 1)))
            train.append(
                evaluate_onnx(
                    onnx_file, arrays, np.sort(rng.choice(train_unique, n_keep, replace=False))
                )
            )
            logger.info(f"Fold {k}: evaluated {len(test_idx)} out-of-fold events")
    finally:
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)

    n_oof = sum(len(t[0]) for t in test)
    if n_oof != n_events:
        raise ValueError(f"{n_oof} out-of-fold scores for {n_events} events")
    logger.info(
        f"Checked that the {k_folds} test folds are disjoint and cover all the "
        f"{n_events} events once"
    )

    merged = {
        "score_lbl_array_test": np.concatenate([t[0] for t in test]),
        "dataset_idx_test": np.concatenate([t[1] for t in test]),
        # fold of each out-of-fold score
        "fold_test": np.concatenate([np.full(len(t[0]), k) for k, t in enumerate(test)]),
        "score_lbl_array_train": np.concatenate([t[0] for t in train]),
        "dataset_idx_train": np.concatenate([t[1] for t in train]),
        # the out-of-fold scores cover all the events
        "train_test_fractions": np.array(
            [sum(len(t[0]) for t in train) / n_events, 1.0]
        ),
        "num_classes": np.array(meta["num_classes"]),
        "class_info": np.array(meta["class_info"], dtype=object),
    }
    if save:
        out_file = f"{out_dir}/score_lbl_array_oof.npz"
        np.savez(out_file, **merged)
        logger.info(f"Saved the out-of-fold scores of {n_oof} events in {out_file}")
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


def run_onnx_all(model_file, inputs):
    """All the outputs of a model."""
    session = ort.InferenceSession(model_file, providers=["CPUExecutionProvider"])
    names = [i.name for i in session.get_inputs()]
    return session.run(None, dict(zip(names, inputs)))


def ratio_of_outputs(out):
    if out.shape[1] == 1:
        return (out / (1 - out)).flatten()
    return out[:, 1] / out[:, 0]


def morphing_weights(fold_out, event, k_folds, alpha, alpha_den):
    """Morphing weights of the 4-node fold outputs computed with numpy, as
    build_multiclass_morphing_models does in ONNX."""
    from ml_pytorch.utils.onnx_tools import RATIO_DEN_EPS

    nums = [o[:, 0] - alpha * o[:, 2] for o in fold_out]
    dens = [o[:, 1] - alpha_den * o[:, 3] for o in fold_out]
    avg_num = np.maximum(np.mean(nums, axis=0), 0)
    avg_den = np.mean(dens, axis=0)
    if alpha_den:
        avg_den = np.maximum(avg_den, 0)
    fold_w = [
        (np.maximum(n, 0) if alpha else n)
        / ((np.maximum(d, 0) if alpha_den else d) + RATIO_DEN_EPS)
        for n, d in zip(nums, dens)
    ]
    return {
        "average_morphing": avg_num / (avg_den + RATIO_DEN_EPS),
        "all_folds_morphing": fold_w,
        "event_morphing": np.choose(event % k_folds, fold_w),
    }


def check_kfold_models(fold_files, kfold_files, n_features, k_folds, morphing=None):
    """Compare the combined models with the per-fold models on random inputs."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(100, n_features)).astype(np.float32)
    # large event numbers, which are not exact in float32
    event = (2**40 + rng.integers(0, 10**6, size=100)).astype(np.int64)
    fold_out = [run_onnx(f, [x]) for f in fold_files]
    # each event with the model of its fold
    selected = np.choose(event % k_folds, [o.T for o in fold_out]).T

    expected = {
        "average": np.mean(fold_out, axis=0),
        "all_folds": fold_out,
        "event": selected,
    }
    if morphing is not None:
        expected.update(morphing_weights(fold_out, event, k_folds, *morphing))
    if "average_ratio" in kfold_files or "event_ratio" in kfold_files:
        fold_ratios = [ratio_of_outputs(o) for o in fold_out]
        expected["average_ratio"] = np.mean(fold_ratios, axis=0)
        expected["event_ratio"] = np.choose(event % k_folds, fold_ratios)
        expected["all_folds_ratio"] = fold_ratios

    for key, model_file in kfold_files.items():
        inputs = [x, event] if key.startswith("event") else [x]
        if key.startswith("all_folds"):
            # one output per fold
            outs = run_onnx_all(model_file, inputs)
            if len(outs) != k_folds or not all(
                np.allclose(o.reshape(e.shape), e, rtol=1e-4, atol=1e-6)
                for o, e in zip(outs, expected[key])
            ):
                raise RuntimeError(
                    f"The combined model {model_file} does not match the fold models"
                )
            logger.info(f"Checked {model_file} against the fold models")
            continue
        out = run_onnx(model_file, inputs)
        # the ratio of a 1-node model keeps the (N, 1) shape of the output
        if out.size != expected[key].size or not np.allclose(
            out.reshape(expected[key].shape), expected[key], rtol=1e-4, atol=1e-6
        ):
            raise RuntimeError(
                f"The combined model {model_file} does not match the fold models"
            )
        logger.info(f"Checked {model_file} against the fold models")


def merge_onnx(
    fold_dirs, out_dir, ratio, event_split, kfold_onnx="average", alpha=None, alpha_den=0.0
):
    """Copy the best model of each fold, build the combined models and save
    the one chosen with kfold_onnx as the final model. With ``alpha`` (4-node
    multiclass morphing models) the final model is the morphing weight with
    the ttbar subtraction (see build_multiclass_morphing_models)."""
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
    morphing = alpha is not None
    if morphing and n_outputs != 4:
        raise ValueError(
            f"alpha is for the 4-node multiclass morphing, the model has {n_outputs} outputs"
        )
    if ratio and n_outputs > 2:
        if not morphing:
            logger.warning(
                f"The ratio of the outputs is defined for 1 or 2 output nodes, the "
                f"model has {n_outputs}: skipping the ratio models (for the "
                f"multiclass morphing give the ttbar fraction alpha)"
            )
        ratio = False

    combined = build_kfold_models(
        onnx_models, n_features, ratio=ratio, event_split=event_split
    )
    if morphing:
        logger.info(f"Multiclass morphing with alpha {alpha}, alpha_den {alpha_den}")
        combined.update(
            build_multiclass_morphing_models(
                onnx_models, n_features, alpha, alpha_den, event_split=event_split
            )
        )
    kfold_files = {}
    for key, model in combined.items():
        kfold_files[key] = f"{out_dir}/{KFOLD_ONNX_NAMES[key]}"
        save_onnx_model(model, kfold_files[key])
    check_kfold_models(
        fold_files,
        kfold_files,
        n_features,
        len(fold_dirs),
        morphing=(alpha, alpha_den) if morphing else None,
    )

    final_key = final_onnx_key(kfold_onnx, ratio, morphing)
    kfold_files["final"] = f"{out_dir}/{final_onnx_name(final_key)}"
    shutil.copy(kfold_files[final_key], kfold_files["final"])
    logger.info(
        f"Final k-fold model (kfold_onnx: {kfold_onnx}, ratio: {ratio}): "
        f"{kfold_files['final']} = {os.path.basename(kfold_files[final_key])}"
    )
    return kfold_files, final_key


def merge_kfold(main_dir, cfg, ratio=None, plots=True, save_numpy=False):
    k_folds = cfg.k_folds
    if not k_folds or k_folds < 2:
        raise ValueError(f"k_folds must be >= 2 to merge the folds, got {k_folds}")
    ratio = cfg.onnx_ratio if ratio is None else ratio
    event_split = cfg.kfold_split == "event"
    if cfg.kfold_onnx == "event" and not event_split:
        raise ValueError("kfold_onnx: event needs kfold_split: event")

    fold_dirs = get_fold_dirs(main_dir, k_folds)
    kfold_dir = f"{main_dir}/kfold"
    os.makedirs(kfold_dir, exist_ok=True)

    check_fold_configs(fold_dirs)
    fold_metrics = [read_fold_metrics(d) for d in fold_dirs]
    summary = summarise_metrics(fold_metrics)
    summary_file = f"{kfold_dir}/kfold_summary.yml"

    merged = merge_scores(
        cfg, fold_dirs, kfold_dir, save=bool(save_numpy or cfg.get("save_numpy", False))
    )

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
            save_roc=bool(cfg.get("save_roc", False)),
        )

    if cfg.onnx:
        alpha = cfg.get("morphing_alpha", None)
        kfold_files, final_key = merge_onnx(
            fold_dirs,
            f"{main_dir}/kfold_onnx",
            ratio,
            event_split,
            cfg.kfold_onnx,
            alpha=alpha,
            alpha_den=cfg.get("morphing_alpha_den", 0.0) or 0.0,
        )
        for key, f in kfold_files.items():
            logger.info(f"K-fold ONNX model ({key}): {f}")
        if alpha is not None:
            summary["morphing_alpha"] = float(alpha)
            summary["morphing_alpha_den"] = float(cfg.get("morphing_alpha_den", 0.0) or 0.0)
        summary["final_onnx"] = {
            "file": kfold_files["final"],
            "kfold_onnx": cfg.kfold_onnx,
            "model": final_key,
            "inputs": (
                ["args_0", "event"] if final_key.startswith("event") else ["args_0"]
            ),
        }

    OmegaConf.save(OmegaConf.create(summary), summary_file)
    logger.info(f"K-fold summary saved in {summary_file}:\n{OmegaConf.to_yaml(summary)}")
    print(OmegaConf.to_yaml(summary))

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
        "--kfold-onnx",
        default=None,
        choices=["average", "event"],
        help="Final ONNX model (overrides config): average over the folds or model of the fold event %% k_folds",
    )
    parser.add_argument(
        "-al",
        "--alpha",
        default=None,
        type=float,
        help="4-node multiclass morphing: ttbar fraction alpha = N_ttbar / N_data of the numerator region (see ml_alpha); the final model is the morphing weight with the ttbar subtraction",
    )
    parser.add_argument(
        "-ald",
        "--alpha-den",
        default=None,
        type=float,
        help="Multiclass morphing: alpha of the denominator region (default 0.0: no ttbar subtraction in the denominator)",
    )
    parser.add_argument(
        "--kfold-cache",
        default=None,
        help="K-fold cache with the events (default: the one of the training if it still exists, otherwise the events are loaded again into a temporary directory)",
    )
    parser.add_argument(
        "-s-n",
        "--save-numpy",
        action="store_true",
        help="Save the out-of-fold scores (kfold/score_lbl_array_oof.npz)",
    )
    parser.add_argument(
        "--save-roc",
        action="store_true",
        help="Save also the ROC points (tpr_fpr*.npz) of the out-of-fold plots",
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
    if args.kfold_onnx is not None:
        cfg.kfold_onnx = args.kfold_onnx
    if args.kfold_cache is not None:
        cfg.kfold_cache = args.kfold_cache
    if args.save_roc:
        cfg.save_roc = True
    if args.alpha is not None:
        cfg.morphing_alpha = args.alpha
    if args.alpha_den is not None:
        cfg.morphing_alpha_den = args.alpha_den
    merge_kfold(
        main_dir, cfg, ratio=args.ratio, plots=not args.no_plots, save_numpy=args.save_numpy
    )


if __name__ == "__main__":
    main()
