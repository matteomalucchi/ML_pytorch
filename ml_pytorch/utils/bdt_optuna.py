"""Optuna search of the BDT hyperparameters, run by ``ml_train`` before the
final training (see ``bdt_tools.run_bdt_training``).

Switched on by an ``optuna:`` block in the config (or ``--optuna N``, the
number of trials)::

    optuna:
      n_trials: 50          # trials to run (a restarted study continues up to this number)
      timeout: 14400        # seconds, optional: stop starting new trials after this time
      seed: 0               # seed of the TPE sampler
      n_startup_trials: 10  # random trials before the TPE sampler is used
      pruner: median        # median or none
      enqueue:              # optional: parameter sets evaluated first (only when the study is new),
        - {max_depth: 6, min_child_weight: 1.0}   # e.g. the baseline, so the search cannot do worse
      storage: null         # default sqlite:///<output_dir>/optuna/study.db
      search_space:         # optional: replaces the default search space
        max_depth: {type: int, low: 3, high: 10}
        learning_rate: {type: log_float, low: 0.01, high: 0.3}
        max_bin: {type: categorical, choices: [64, 128, 256]}

The objective is the weighted validation metric (``eval_param``: mlogloss or
merror) at the best round of each trial: the test events are not used. Every
trial is an early-stopped training on the same training and validation events,
so ``epochs`` is the maximum number of rounds. The training events are the
same for all the trials, with the same weights as in the final training. The
final training is the normal one with the best parameters.
"""

import logging
import os
import time

import numpy as np
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

DEFAULT_SEARCH_SPACE = {
    "max_depth": {"type": "int", "low": 3, "high": 10},
    "learning_rate": {"type": "log_float", "low": 0.01, "high": 0.3},
    "min_child_weight": {"type": "log_float", "low": 1.0, "high": 100.0},
    "subsample": {"type": "float", "low": 0.5, "high": 1.0},
    "colsample_bytree": {"type": "float", "low": 0.5, "high": 1.0},
    "reg_lambda": {"type": "log_float", "low": 1e-2, "high": 100.0},
    "reg_alpha": {"type": "log_float", "low": 1e-3, "high": 10.0},
    "gamma": {"type": "float", "low": 0.0, "high": 5.0},
    "max_bin": {"type": "categorical", "choices": [64, 128, 256]},
}
DEFAULT_OPTUNA = {
    "n_trials": 50,
    "timeout": None,
    "seed": 0,
    "n_startup_trials": 10,
    "pruner": "median",
    "enqueue": None,
    "storage": None,
    "search_space": None,
}


def get_optuna_config(cfg):
    """The ``optuna`` block of the config with the defaults filled in
    (``--optuna N`` sets n_trials), or None if the search is off."""
    block = cfg.get("optuna", None)
    n_trials = cfg.get("optuna_trials", None)
    if not block and not n_trials:
        return None
    config = dict(DEFAULT_OPTUNA)
    if block:
        config.update(OmegaConf.to_container(block, resolve=True))
    if n_trials:
        config["n_trials"] = int(n_trials)
    if config["n_trials"] < 1:
        raise ValueError("optuna n_trials must be >= 1")
    return config


def suggest(trial, name, spec):
    kind = spec["type"]
    if kind == "int":
        return trial.suggest_int(name, spec["low"], spec["high"], step=spec.get("step", 1))
    if kind == "float":
        return trial.suggest_float(name, spec["low"], spec["high"])
    if kind == "log_float":
        return trial.suggest_float(name, spec["low"], spec["high"], log=True)
    if kind == "categorical":
        return trial.suggest_categorical(name, list(spec["choices"]))
    raise ValueError(f"Unknown type {kind} of the Optuna parameter {name}")


def pruning_callback(trial, metric, data_name):
    """XGBoost callback reporting the validation metric to Optuna at every
    round and stopping the trial if the pruner says so."""
    import optuna
    import xgboost as xgb

    class PruningCallback(xgb.callback.TrainingCallback):
        def after_iteration(self, model, epoch, evals_log):
            trial.report(float(evals_log[data_name][metric][-1]), epoch)
            if trial.should_prune():
                raise optuna.TrialPruned(f"pruned at round {epoch}")
            return False

    return PruningCallback()


def run_optuna_search(cfg, ML_model, train, val, num_classes, input_variables, main_dir, config):
    """Run the study. Returns (best parameters, info for training_info.yml);
    writes ``optuna/best_params.yml``, ``optuna/trials.csv``, the study
    database and the plots in ``<main_dir>/optuna``."""
    try:
        import optuna
    except ImportError as e:
        raise ImportError("The hyperparameter search needs optuna: pip install optuna") from e
    from ml_pytorch.models.XGB_multiclass_model import (
        VALIDATION_NAME,
        eval_metric_name,
    )
    from ml_pytorch.utils.bdt_tools import fit_bdt

    directory = f"{main_dir}/optuna"
    os.makedirs(directory, exist_ok=True)
    metric = eval_metric_name(cfg.eval_param)
    space = config["search_space"] or DEFAULT_SEARCH_SPACE
    storage = config["storage"] or f"sqlite:///{directory}/study.db"
    pruner = (
        optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=20)
        if config["pruner"] == "median"
        else optuna.pruners.NopPruner()
    )
    study = optuna.create_study(
        study_name=os.path.basename(main_dir.rstrip("/")),
        storage=storage,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            seed=int(config["seed"]), n_startup_trials=int(config["n_startup_trials"])
        ),
        pruner=pruner,
        load_if_exists=True,
    )
    if config.get("enqueue") and len(study.trials) == 0:
        for params in config["enqueue"]:
            study.enqueue_trial(dict(params))
        logger.info(f"Optuna: enqueued {len(config['enqueue'])} parameter sets to evaluate first")
    finished = [
        t
        for t in study.trials
        if t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
    ]
    remaining = int(config["n_trials"]) - len(finished)
    logger.info(
        f"Optuna: {len(finished)} finished trials in the study, running {max(remaining, 0)} more "
        f"(objective: validation {metric}, search space: {list(space)})"
    )

    def objective(trial):
        params = {name: suggest(trial, name, spec) for name, spec in space.items()}
        model = ML_model.get_model(
            num_classes,
            cfg,
            params=params,
            extra_callbacks=[pruning_callback(trial, metric, VALIDATION_NAME)],
        )
        start = time.time()
        fit_bdt(model, train, val)
        scores = model.evals_result()[VALIDATION_NAME][metric]
        trial.set_user_attr("best_round", int(model.best_iteration))
        trial.set_user_attr("time_s", time.time() - start)
        return float(np.min(scores))

    def log_trial(study, trial):
        logger.info(
            "Optuna trial %d: %s, value %s, params %s"
            % (trial.number, trial.state.name, trial.value, trial.params)
        )

    if remaining > 0:
        study.optimize(
            objective,
            n_trials=remaining,
            timeout=config["timeout"],
            callbacks=[log_trial],
            gc_after_trial=True,
        )

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise RuntimeError("Optuna: no trial completed, no best parameters")
    best = study.best_trial
    best_params = dict(best.params)
    logger.info(
        f"Optuna best trial {best.number}: {metric} {best.value:.5f}, best round "
        f"{best.user_attrs.get('best_round')}, parameters {best_params}"
    )

    OmegaConf.save(
        OmegaConf.create(
            {
                "best_trial": best.number,
                "objective": f"validation {metric}",
                "value": float(best.value),
                "best_round": best.user_attrs.get("best_round"),
                "params": best_params,
            }
        ),
        f"{directory}/best_params.yml",
    )
    study.trials_dataframe().to_csv(f"{directory}/trials.csv", index=False)
    plot_study(study, directory, metric)

    info = {
        "n_trials": len(study.trials),
        "n_complete": len(completed),
        "n_pruned": sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials),
        "best_trial": best.number,
        "best_value": float(best.value),
        "objective": f"validation {metric}",
    }
    return best_params, info


def plot_study(study, directory, metric):
    """Optimisation history, parameter importances and parameter-vs-value
    scatter plots of the study (matplotlib, saved as png/pdf/svg)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mplhep as hep
    import optuna

    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    def save(fig, name):
        for ext in ["png", "pdf", "svg"]:
            fig.savefig(f"{directory}/{name}.{ext}")
        plt.close(fig)

    with plt.style.context(hep.style.CMS):
        fig, ax = plt.subplots(figsize=(10, 7))
        numbers = [t.number for t in complete]
        values = [t.value for t in complete]
        ax.plot(numbers, values, "o", label="trial")
        ax.plot(numbers, np.minimum.accumulate(values), "-", label="best so far")
        ax.set_xlabel("Trial")
        ax.set_ylabel(f"Validation {metric}")
        ax.legend()
        save(fig, "optimization_history")

        try:
            importances = optuna.importance.get_param_importances(study)
        except Exception as e:  # too few trials, or constant parameters
            logger.warning(f"No Optuna parameter importances: {e}")
            importances = {}
        if importances:
            fig, ax = plt.subplots(figsize=(10, 6))
            names = list(importances)[::-1]
            ax.barh(names, [importances[n] for n in names])
            ax.set_xlabel("Importance for the objective")
            save(fig, "param_importances")

        params = sorted({p for t in complete for p in t.params})
        if params:
            ncols = 3
            nrows = int(np.ceil(len(params) / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
            for ax, name in zip(axes.flat, params):
                ax.scatter(
                    [t.params[name] for t in complete if name in t.params],
                    [t.value for t in complete if name in t.params],
                    s=12,
                )
                ax.set_xlabel(name)
                ax.set_ylabel(f"Validation {metric}")
            for ax in list(axes.flat)[len(params):]:
                ax.axis("off")
            fig.tight_layout()
            save(fig, "param_vs_value")
