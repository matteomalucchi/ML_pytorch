"""XGBoost multi-class BDT, the counterpart of DNN_multiclass_model.

Unlike the DNN models this module does not define a torch model: ``ml_train``
sees ``IS_BDT = True`` and hands the training to
``ml_pytorch.utils.bdt_tools.run_bdt_training``, which reuses the data loading,
the class weights, the logging format, the evaluation plots and the output
folder layout of the DNN training.

Mapping of the common config keys:

    epochs          -> n_estimators (maximum number of trees per class)
    learning_rate   -> learning_rate (eta), unless ``bdt_params.learning_rate`` is set
    patience        -> early stopping rounds (all the rounds if ``early_stopping: False``)
    min_delta       -> minimum improvement of the validation metric
    eval_param      -> loss: validation mlogloss, acc: validation merror
    seed            -> random_state
    gpus            -> device ("cpu" or the first GPU number -> "cuda:<n>")

Everything else is set in the ``bdt_params`` dictionary of the config, which
overrides DEFAULT_PARAMS (any XGBClassifier argument is accepted).

The model outputs the probabilities of the classes (multi:softprob), as the
softmax of the DNN models.
"""

import xgboost as xgb
from omegaconf import OmegaConf

IS_BDT = True

DEFAULT_PARAMS = {
    "max_depth": 6,
    "min_child_weight": 1.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "gamma": 0.0,
    "max_bin": 256,
    "tree_method": "hist",
}

# data_name of the validation events in the eval_set of the fit:
# [training events, validation events]
VALIDATION_NAME = "validation_1"
TRAINING_NAME = "validation_0"


def resolve_device(gpus):
    """XGBoost device from the ``gpus`` option of ml_train."""
    if gpus is None or str(gpus).strip().lower() == "cpu":
        return "cpu"
    return f"cuda:{int(str(gpus).split(',')[0])}"


def eval_metric_name(eval_param):
    """Validation metric that selects the best iteration."""
    if eval_param == "acc":
        return "merror"
    if eval_param == "loss":
        return "mlogloss"
    raise ValueError(f"eval_param must be loss or acc, got {eval_param}")


def get_params(cfg, params=None):
    """Hyperparameters of the BDT: DEFAULT_PARAMS < ``bdt_params`` of the
    config < ``params`` (e.g. the ones sampled by Optuna). The learning rate
    comes from ``learning_rate`` unless one of them sets it."""
    out = dict(DEFAULT_PARAMS)
    bdt_params = cfg.get("bdt_params", None)
    if bdt_params:
        out.update(OmegaConf.to_container(bdt_params, resolve=True))
    if params:
        out.update(params)
    out.setdefault("learning_rate", float(cfg.learning_rate))
    return out


def get_model(num_classes, cfg, params=None, extra_callbacks=None):
    """Untrained XGBClassifier. Fit it with the eval_set
    [(training), (validation)]: the early stopping follows the validation
    events and, with ``save_best``, the trees after the best iteration are
    removed at the end of the fit (needed for the ONNX export)."""
    p = get_params(cfg, params)
    metric = eval_metric_name(cfg.eval_param)
    early_stopping = xgb.callback.EarlyStopping(
        rounds=int(cfg.patience) if cfg.early_stopping else int(cfg.epochs),
        min_delta=float(cfg.min_delta),
        metric_name=metric,
        data_name=VALIDATION_NAME,
        save_best=True,
    )
    model = xgb.XGBClassifier(
        objective="multi:softprob",
        n_estimators=int(cfg.epochs),
        eval_metric=["mlogloss", "merror"],
        device=resolve_device(cfg.get("gpus", None)),
        random_state=int(cfg.seed),
        callbacks=[early_stopping, *(extra_callbacks or [])],
        **p,
    )
    return model
