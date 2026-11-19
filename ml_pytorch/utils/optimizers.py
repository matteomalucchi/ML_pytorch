"""Optimizers selectable from the config (optimizer, weight_decay), passed by
ml_train to the get_model of every model, as the learning rate schedule."""
import torch

# name in the config -> torch optimizer
OPTIMIZERS = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
}


def check_optimizer_cfg(name, weight_decay):
    """Check the optimizer and weight_decay of the config (before loading the
    data)."""
    if name not in OPTIMIZERS:
        raise ValueError(
            f"Invalid optimizer {name}: use one of {list(OPTIMIZERS)}"
        )
    if weight_decay is not None and float(weight_decay) < 0:
        raise ValueError(f"weight_decay must be >= 0 or null, got {weight_decay}")


def get_optimizer(name, params, lr, weight_decay=None):
    """Optimizer ``name`` (adam or adamw) of the parameters ``params`` with
    learning rate ``lr``. ``weight_decay`` is passed only if not None,
    otherwise the PyTorch default of the optimizer is used (0 for Adam, 0.01
    for AdamW). For Adam it is the coefficient of the L2 penalty (added to the
    gradient), for AdamW the decoupled weight decay."""
    check_optimizer_cfg(name, weight_decay)
    kwargs = {}
    if weight_decay is not None:
        kwargs["weight_decay"] = float(weight_decay)
    return OPTIMIZERS[name](params, lr=lr, **kwargs)
