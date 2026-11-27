"""Training, evaluation and outputs of the BDT (XGBoost) of ``ml_train``.

Counterpart of ``utils/tools.py`` (torch loop) for the models with
``IS_BDT = True`` (see ``models/XGB_multiclass_model.py``). The data loading
(``load_data``), class weights, evaluation plots (``plot_kl_distributions``)
and history plots are the ones of the DNN training: the BDT writes its
training history in the same log format as the DNN, and its scores in the same
``score_lbl_array`` layout, so that the output folder looks like the one of a
DNN training. Specific to the BDT: ``feature_importance/``, ``training_info.yml``
and, with Optuna, ``optuna/``.
"""

import importlib
import json
import logging
import os
import time

import numpy as np
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

# the probabilities of the BDT in the ONNX file, as for the DNN models
ONNX_INPUT_NAME = "InputVariables"
ONNX_OUTPUT_NAME = "Output"
# IR version understood by the onnxruntime of the pocket-coffea environment
# (1.19: up to IR 10), while the onnx package writes a newer one
ONNX_IR_VERSION = 8
# probabilities of the ONNX model and of the BDT must agree to this precision
ONNX_MAX_ABS_DIFF = 1e-5
IMPORTANCE_TYPES = ["gain", "total_gain", "weight", "cover"]


def is_bdt_model(model_name):
    """True if the model module ``ml_pytorch.models.<model_name>`` is a BDT."""
    try:
        module = importlib.import_module(f"ml_pytorch.models.{model_name}")
    except ImportError:
        return False
    return bool(getattr(module, "IS_BDT", False))


def check_bdt_cfg(cfg):
    """Options that do not exist for the BDT (one model per training)."""
    if (cfg.get("k_folds", 0) or 0) >= 2:
        raise NotImplementedError("k-fold training is not supported for the BDT")
    if cfg.get("load_model", None):
        raise NotImplementedError(
            "--load-model (resuming) is not supported for the BDT; "
            "use --eval-model <model_best_epoch_N.json> to evaluate a trained BDT"
        )
    if cfg.get("oversample_split", False) or cfg.get("split_oversample", False):
        logger.warning(
            "Oversampling repeats events: the BDT balances the classes with weights, "
            "set oversample_split: False"
        )


def to_numpy(array):
    if hasattr(array, "detach"):
        array = array.detach().cpu()
    if hasattr(array, "numpy"):
        array = array.numpy()
    return np.asarray(array)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def loader_to_arrays(loader):
    """All the events of a loader of ``load_data`` as numpy arrays.

    ``w`` is the training weight, event weight x class weight clamped at 0,
    as in ``loop_one_batch``. ``event_w`` is the event weight alone (the one
    saved in the score arrays). The standard loaders drop the last incomplete
    batch of the test events, here all the events are used.
    """
    features, labels, class_weights, kl_info = [to_numpy(t) for t in loader.dataset[:]]
    event_w = features[:, -1].astype(np.float64)
    labels = labels.reshape(-1).astype(np.int64)
    return {
        # the last column of the features is the event weight
        "X": np.ascontiguousarray(features[:, :-1], dtype=np.float32),
        "y": labels,
        "event_w": event_w,
        "w": np.clip(event_w * class_weights.reshape(-1).astype(np.float64), 0, None),
        "kl": kl_info[:, 0].astype(np.float64),
        "dataset_idx": kl_info[:, 1].astype(np.int64),
    }


def check_labels(arrays, num_classes):
    """The labels must be 0 ... num_classes - 1 (they are the output nodes)."""
    present = np.unique(arrays["y"])
    if present.min() < 0 or present.max() >= num_classes:
        raise ValueError(
            f"Labels {present.tolist()} are not in 0 ... {num_classes - 1}"
        )


def weighted_metrics(proba, y, w):
    """Weighted multi-class log loss and accuracy."""
    p = np.clip(proba[np.arange(len(y)), y], 1e-15, 1.0)
    sum_w = w.sum()
    loss = float(-(w * np.log(p)).sum() / sum_w)
    accuracy = float(((proba.argmax(axis=1) == y) * w).sum() / sum_w)
    return loss, accuracy


def eval_bdt(model, arrays):
    """Counterpart of ``tools.eval_model``: ``(score_lbl_array,
    dataset_idx, loss, accuracy)``, with ``score_lbl_array`` of the layout
    [class probabilities..., label, event weight, kl]."""
    proba = model.predict_proba(arrays["X"])
    loss, accuracy = weighted_metrics(proba, arrays["y"], arrays["w"])
    score_lbl_array = np.concatenate(
        [
            proba,
            arrays["y"][:, None],
            arrays["event_w"][:, None],
            arrays["kl"][:, None],
        ],
        axis=1,
    ).astype(np.float32)
    return score_lbl_array, arrays["dataset_idx"], loss, accuracy


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def progress_callback(every=10):
    """XGBoost callback logging the validation metrics every ``every`` rounds."""
    import xgboost as xgb

    class ProgressLogger(xgb.callback.TrainingCallback):
        def __init__(self):
            self.start = time.time()

        def after_iteration(self, model, epoch, evals_log):
            if epoch % every == 0:
                values = ", ".join(
                    f"{data} {metric} {scores[-1]:.5f}"
                    for data, metrics in evals_log.items()
                    for metric, scores in metrics.items()
                )
                logger.info(
                    "Round %d (%.0f s): %s" % (epoch, time.time() - self.start, values)
                )
            return False

    return ProgressLogger()


def fit_bdt(model, train, val):
    """Fit with the training events and the weights, early stopping on the
    validation events. Returns the time spent."""
    start = time.time()
    model.fit(
        train["X"],
        train["y"],
        sample_weight=train["w"],
        eval_set=[(train["X"], train["y"]), (val["X"], val["y"])],
        sample_weight_eval_set=[train["w"], val["w"]],
        verbose=False,
    )
    return time.time() - start


def history_from_model(model):
    """Per-round weighted loss and accuracy of the training and validation
    events (all the rounds that were run, also the ones after the best)."""
    from ml_pytorch.models.XGB_multiclass_model import TRAINING_NAME, VALIDATION_NAME

    result = model.evals_result()
    history = {}
    for key, name in [("train", TRAINING_NAME), ("val", VALIDATION_NAME)]:
        history[f"{key}_loss"] = np.asarray(result[name]["mlogloss"])
        history[f"{key}_accuracy"] = 1.0 - np.asarray(result[name]["merror"])
    return history


def log_history(history, learning_rate, best_iteration, comet_logger=None):
    """Write the history of the rounds in the format of the DNN log, so that
    ``plot_history.read_from_txt`` and the k-fold summary read it unchanged.
    One round is one epoch."""
    n_rounds = len(history["val_loss"])
    best_val_loss, best_val_accuracy, best_round = np.inf, 0.0, -1
    for r in range(n_rounds):
        for name, kind in [("Training", "train"), ("Validation", "val")]:
            # a DNN epoch logs 10 progress lines (every 10% of the batches) and
            # plot_history takes 10 points per epoch: write the value of the
            # round 10 times so that the x axis of the plot is in rounds
            for percent in range(10, 101, 10):
                logger.info(
                    "EPOCH # %d, time 0.0,  %s batch %.1f %% , step %d        "
                    "accuracy: %.4f      //      loss: %.4f"
                    % (r, name, percent, r + 1, history[f"{kind}_accuracy"][r], history[f"{kind}_loss"][r])
                )
            if kind == "train":
                logger.info("EPOCH # %d, learning rate: %.6f" % (r, learning_rate))
        if history["val_loss"][r] < best_val_loss:
            best_val_loss = history["val_loss"][r]
            best_val_accuracy = history["val_accuracy"][r]
            best_round = r
        logger.info(
            "EPOCH # %d: loss train %.4f,  val %.4f"
            % (r, history["train_loss"][r], history["val_loss"][r])
        )
        logger.info(
            "EPOCH # %d: acc train %.4f,  val %.4f"
            % (r, history["train_accuracy"][r], history["val_accuracy"][r])
        )
        logger.info(
            "Best epoch # %d, best val loss: %.4f, best val accuracy: %.4f"
            % (best_round, best_val_loss, best_val_accuracy)
        )
        if comet_logger:
            for key, value in history.items():
                comet_logger.log_metric(f"{key}_epoch", float(value[r]), epoch=r)
    logger.info(
        "Best round: # %d of %d rounds run (the trees after it are removed)"
        % (best_iteration, n_rounds)
    )


# ---------------------------------------------------------------------------
# ONNX
# ---------------------------------------------------------------------------


def check_onnx_dependencies():
    """Fail before the training if the ONNX export cannot work."""
    try:
        import onnxmltools  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "The ONNX export of the BDT needs onnxmltools: pip install onnxmltools "
            "(or train without --onnx)"
        ) from e


def _rename_tensor(model, old, new):
    for node in model.graph.node:
        for i, name in enumerate(node.input):
            if name == old:
                node.input[i] = new
        for i, name in enumerate(node.output):
            if name == old:
                node.output[i] = new
    for value in list(model.graph.output) + list(model.graph.input):
        if value.name == old:
            value.name = new


def export_bdt_onnx(model, n_features, onnx_file, test_inputs=None):
    """Save the BDT as an ONNX model with the interface of the DNN models:
    input ``InputVariables`` [N, n_features] float32, output ``Output``
    [N, num_classes] probabilities. The probabilities are checked against the
    BDT on ``test_inputs`` (max abs diff < ONNX_MAX_ABS_DIFF)."""
    import onnx
    import onnxruntime as ort
    from onnxmltools import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType

    converted = convert_xgboost(
        model,
        initial_types=[(ONNX_INPUT_NAME, FloatTensorType([None, n_features]))],
    )
    outputs = [o.name for o in converted.graph.output]
    logger.info(f"Outputs of the converted BDT: {outputs}")
    # keep only the probabilities (the converter also outputs the label)
    proba_name = next(n for n in outputs if "prob" in n.lower())
    extractor = onnx.utils.Extractor(converted)
    onnx_model = extractor.extract_model([ONNX_INPUT_NAME], [proba_name])
    _rename_tensor(onnx_model, proba_name, ONNX_OUTPUT_NAME)
    onnx_model.ir_version = ONNX_IR_VERSION
    onnx.checker.check_model(onnx_model)
    os.makedirs(os.path.dirname(onnx_file), exist_ok=True)
    onnx.save(onnx_model, onnx_file)
    logger.info(f"Saved the BDT as ONNX model: {onnx_file}")

    if test_inputs is not None:
        session = ort.InferenceSession(onnx_file, providers=["CPUExecutionProvider"])
        onnx_proba = session.run(None, {ONNX_INPUT_NAME: test_inputs.astype(np.float32)})[0]
        diff = float(np.abs(onnx_proba - model.predict_proba(test_inputs)).max())
        logger.info(f"Max |ONNX - BDT| probability on {len(test_inputs)} events: {diff:.2e}")
        if diff > ONNX_MAX_ABS_DIFF:
            raise RuntimeError(
                f"The ONNX model does not reproduce the BDT (max abs diff {diff:.2e} "
                f"> {ONNX_MAX_ABS_DIFF:.0e}): {onnx_file}"
            )
        return diff
    return None


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------


def importance_tables(model, input_variables, class_names):
    """Importance of the input variables from the splits of the trees.

    Returns {"all": {type: {variable: value}}, "<class>": {...}}, with the
    types gain (mean gain per split), total_gain, weight (number of splits)
    and cover (mean cover per split). The trees of a multi-class BDT are
    ordered by round and then by class, so tree ``t`` is of class
    ``t % num_classes``: the importance per class comes from its own trees.
    """
    frame = model.get_booster().trees_to_dataframe()
    frame = frame[frame["Feature"] != "Leaf"].copy()
    num_classes = len(class_names)
    frame["class_idx"] = frame["Tree"] % num_classes
    # without feature names the booster calls the columns f0, f1, ...
    index = frame["Feature"].str.lstrip("f").astype(int)
    frame["variable"] = [input_variables[i] for i in index]

    def table(sub):
        grouped = sub.groupby("variable")
        values = {
            "gain": grouped["Gain"].mean(),
            "total_gain": grouped["Gain"].sum(),
            "weight": grouped["Gain"].count().astype(float),
            "cover": grouped["Cover"].mean(),
        }
        return {
            kind: {v: float(values[kind].get(v, 0.0)) for v in input_variables}
            for kind in IMPORTANCE_TYPES
        }

    tables = {"all": table(frame)}
    for idx, name in enumerate(class_names):
        tables[name] = table(frame[frame["class_idx"] == idx])
    return tables


def plot_importance(values, title, out_name, top=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mplhep as hep

    items = sorted(values.items(), key=lambda kv: kv[1])
    if top:
        items = items[-top:]
    names = [k for k, _ in items]
    numbers = [v for _, v in items]
    with plt.style.context(hep.style.CMS):
        fig, ax = plt.subplots(figsize=(10, max(5, 0.28 * len(names) + 2)))
        ax.barh(range(len(names)), numbers, color=hep.style.CMS["axes.prop_cycle"].by_key()["color"][0])
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=10)
        ax.set_xlabel(title)
        fig.tight_layout()
        for ext in ["png", "pdf", "svg"]:
            fig.savefig(f"{out_name}.{ext}")
        plt.close(fig)


def feature_importance(model, input_variables, class_names, out_dir):
    """Write ``feature_importance/feature_importance.yml`` (all the numbers) and
    the plots of the fraction of the total gain, for all the classes and for
    each class (``feature_importance_gain*.png/pdf/svg``)."""
    directory = f"{out_dir}/feature_importance"
    os.makedirs(directory, exist_ok=True)
    tables = importance_tables(model, input_variables, class_names)
    OmegaConf.save(OmegaConf.create(tables), f"{directory}/feature_importance.yml")
    for name, table in tables.items():
        total = sum(table["total_gain"].values())
        fraction = {v: g / total for v, g in table["total_gain"].items()} if total > 0 else table["total_gain"]
        suffix = "" if name == "all" else f"_class_{name}"
        label = "all classes" if name == "all" else f"class {name}"
        plot_importance(
            fraction,
            f"Fraction of the total gain ({label})",
            f"{directory}/feature_importance_gain{suffix}",
        )
        plot_importance(
            table["weight"],
            f"Number of splits ({label})",
            f"{directory}/feature_importance_splits{suffix}",
        )
    logger.info(f"Saved the feature importance in {directory}")
    return tables


# ---------------------------------------------------------------------------
# Main entry point called by ml_train
# ---------------------------------------------------------------------------


def run_bdt_training(
    cfg,
    ML_model,
    loaders,
    num_classes,
    class_info,
    input_variables,
    main_dir,
    lumitext,
    comet_logger,
    save_numpy,
):
    """Training, evaluation and all the outputs of a BDT. Called by ml_train
    after load_data and the plots of the input variables; ``loaders`` is
    (training, validation, test) as returned by load_data."""
    start_time = time.time()
    training_loader, val_loader, test_loader = loaders
    check_bdt_cfg(cfg)
    if cfg.onnx:
        check_onnx_dependencies()

    train = loader_to_arrays(training_loader)
    check_labels(train, num_classes)
    val = loader_to_arrays(val_loader) if val_loader is not None else None
    test = loader_to_arrays(test_loader) if test_loader is not None else None
    n_features = train["X"].shape[1]
    if n_features != len(input_variables):
        raise ValueError(
            f"{n_features} feature columns for {len(input_variables)} input variables"
        )
    logger.info(
        "Events: training %d, validation %d, test %d, features %d"
        % (len(train["y"]), len(val["y"]) if val else 0, len(test["y"]) if test else 0, n_features)
    )

    optuna_info = None
    if cfg.eval_model:
        import xgboost as xgb

        model = xgb.XGBClassifier()
        model.load_model(cfg.eval_model)
        best_iteration = int(os.path.basename(cfg.eval_model).split("best_epoch_")[1].split(".")[0]) if "best_epoch_" in cfg.eval_model else -1
        history = None
        train_time = 0.0
    else:
        from ml_pytorch.utils.bdt_optuna import get_optuna_config, run_optuna_search

        optuna_config = get_optuna_config(cfg)
        if optuna_config:
            logger.info("Optuna search of the BDT hyperparameters")
            best_params, optuna_info = run_optuna_search(
                cfg, ML_model, train, val, num_classes, input_variables, main_dir, optuna_config
            )
            cfg.bdt_params = OmegaConf.create(
                {**(OmegaConf.to_container(cfg.bdt_params) if cfg.get("bdt_params") else {}), **best_params}
            )
            OmegaConf.save(cfg, f"{main_dir}/config_parameters.yml")
            logger.info(f"Training the final BDT with the best parameters: {best_params}")

        params = ML_model.get_params(cfg)
        logger.info(f"BDT parameters: {params}")
        model = ML_model.get_model(num_classes, cfg, extra_callbacks=[progress_callback()])
        train_time = fit_bdt(model, train, val)
        best_iteration = int(model.best_iteration)
        history = history_from_model(model)
        logger.info(
            "Trained %d trees per class in %.1f s, best round # %d"
            % (best_iteration + 1, train_time, best_iteration)
        )
        log_history(history, params["learning_rate"], best_iteration, comet_logger)

    state_dir = f"{main_dir}/state_dict"
    os.makedirs(state_dir, exist_ok=True)
    if not cfg.eval_model:
        model.save_model(f"{state_dir}/model_best_epoch_{best_iteration}.json")

    if cfg.history and history is not None:
        from ml_pytorch.scripts.plot_history import plot_history, plot_lr, read_from_txt

        logger.info("Plotting training and validation loss and accuracy")
        log_files = [h.baseFilename for h in logging.getLogger().handlers if hasattr(h, "baseFilename")]
        tr_acc, tr_loss, va_acc, va_loss, lr = read_from_txt(log_files[0])
        plot_history(tr_acc, tr_loss, va_acc, va_loss, main_dir, False, comet_logger=comet_logger, lumitext=lumitext)
        plot_lr(lr, main_dir, False, comet_logger=comet_logger, lumitext=lumitext)

    if cfg.onnx:
        logger.info("Exporting model to ONNX")
        onnx_name = (
            f"model_best_epoch_{best_iteration}"
            if not cfg.eval_model
            else os.path.basename(cfg.eval_model).replace(".json", "")
        )
        reference = (test if test is not None else val)["X"][:5000]
        export_bdt_onnx(model, n_features, f"{state_dir}/{onnx_name}.onnx", reference)

    class_names = [c["name"] for c in class_info]
    importance = feature_importance(model, input_variables, class_names, main_dir)

    metrics = {}
    if cfg.eval or cfg.eval_model:
        logger.info("Evaluating best model on test and train dataset")
        score_train, dataset_idx_train, loss_train, acc_train = eval_bdt(model, train)
        score_test, dataset_idx_test, loss_test, acc_test = eval_bdt(model, test)
        if val is not None:
            loss_val, acc_val = weighted_metrics(model.predict_proba(val["X"]), val["y"], val["w"])
            logger.info("Best epoch # %d, loss val: %.4f, accuracy val: %.4f" % (best_iteration, loss_val, acc_val))
            metrics.update(val_loss=loss_val, val_accuracy=acc_val)
        logger.info(
            "Eval epoch # %d,  loss train: %.4f,  loss test: %.4f" % (best_iteration, loss_train, loss_test)
        )
        logger.info(
            "Eval epoch # %d,  acc train: %.4f,  acc test: %.4f" % (best_iteration, acc_train, acc_test)
        )
        metrics.update(
            train_loss=loss_train, test_loss=loss_test, train_accuracy=acc_train, test_accuracy=acc_test
        )
        train_test_fractions = np.array([cfg.train_fraction, cfg.test_fraction])

        if save_numpy:
            np.savez(
                f"{main_dir}/score_lbl_array.npz",
                score_lbl_array_train=score_train,
                score_lbl_array_test=score_test,
                dataset_idx_train=dataset_idx_train,
                dataset_idx_test=dataset_idx_test,
                train_test_fractions=train_test_fractions,
                num_classes=np.array(num_classes),
                class_info=np.array(class_info, dtype=object),
            )

        if cfg.histos or cfg.roc:
            from ml_pytorch.scripts.sig_bkg_eval import plot_kl_distributions_with_fallback

            plot_kl_distributions_with_fallback(
                score_train,
                score_test,
                main_dir,
                train_test_fractions[1],
                do_histos=cfg.histos,
                do_roc=cfg.roc,
                comet_logger=comet_logger,
                class_info=class_info,
                lumitext=lumitext,
                dataset_idx_train=dataset_idx_train,
                dataset_idx_test=dataset_idx_test,
            )

    write_training_info(
        main_dir, cfg, ML_model, train, val, test, best_iteration, metrics,
        train_time, time.time() - start_time, class_info, optuna_info, importance,
    )
    logger.info("Saved output in %s" % main_dir)
    logger.info("Total time: %.0f s" % (time.time() - start_time))


def write_training_info(
    main_dir, cfg, ML_model, train, val, test, best_iteration, metrics,
    train_time, total_time, class_info, optuna_info, importance,
):
    """``training_info.yml``: parameters, sizes, metrics and times."""
    def sizes(arrays):
        if arrays is None:
            return None
        return {
            "events": int(len(arrays["y"])),
            "sum_event_weights": float(arrays["event_w"].sum()),
            "sum_training_weights": float(arrays["w"].sum()),
            "events_per_class": {
                c["name"]: int((arrays["y"] == c["lbl"]).sum()) for c in class_info
            },
        }

    top = sorted(importance["all"]["total_gain"].items(), key=lambda kv: -kv[1])[:10]
    info = {
        "model": cfg.ML_model,
        "parameters": ML_model.get_params(cfg),
        "best_round": int(best_iteration),
        "trees_per_class": int(best_iteration) + 1,
        "n_features": int(train["X"].shape[1]),
        "metrics": metrics,
        "datasets": {"training": sizes(train), "validation": sizes(val), "test": sizes(test)},
        "top_features_by_total_gain": [name for name, _ in top],
        "training_time_s": float(train_time),
        "total_time_s": float(total_time),
        "optuna": optuna_info,
    }
    OmegaConf.save(OmegaConf.create(json.loads(json.dumps(info, default=float))), f"{main_dir}/training_info.yml")
