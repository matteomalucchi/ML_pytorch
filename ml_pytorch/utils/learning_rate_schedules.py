import math
import re

from torch.optim import Optimizer
from torch.optim.lr_scheduler import ConstantLR, LinearLR, LambdaLR

# eX_dropYY: constant for X epochs, then multiplied by 0.YY every epoch
DELAYED_DROP_PATTERN = re.compile(r"e(\d+)_drop(\d+)")
# wX_cosineYY: linear warm-up for X epochs, then cosine decay down to
# learning_rate * 0.YY at the last epoch (wX_cosine: down to 0)
WARMUP_COSINE_PATTERN = re.compile(r"w(\d+)_cosine(\d*)")


def get_constant_schedule(optimizer: Optimizer):
    return ConstantLR(optimizer, factor=1.0)


def get_linear_schedule_with_warmup(optimizer: Optimizer, total_iters: int):
    return LinearLR(
        optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_iters
    )

def get_delayed_drop_schedule(optimizer: Optimizer, delay_epochs: int, drop_factor: float):
    decay_func = lambda epoch: (drop_factor**(epoch-delay_epochs) if epoch>=delay_epochs else 1)
    return LambdaLR(
        optimizer, lr_lambda=decay_func
    )


def get_warmup_cosine_schedule(
    optimizer: Optimizer, warmup_epochs: int, min_factor: float, n_epochs: int
):
    """Factor (epoch + 1) / warmup_epochs in the first warmup_epochs epochs,
    then cosine decay from 1 at epoch warmup_epochs to min_factor at the last
    epoch (n_epochs - 1)."""
    decay_epochs = max(n_epochs - 1 - warmup_epochs, 1)

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = min((epoch - warmup_epochs) / decay_epochs, 1.0)
        return min_factor + (1 - min_factor) * 0.5 * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def parse_warmup_cosine(lr_schedule):
    """(warmup_epochs, min_factor) of a schedule named wX_cosineYY, e.g.
    w5_cosine01 -> (5, 0.01), w10_cosine -> (10, 0.0). None if lr_schedule
    does not have this form."""
    match = WARMUP_COSINE_PATTERN.fullmatch(lr_schedule)
    if match is None:
        return None
    warmup_epochs = int(match.group(1))
    min_factor = float(f"0.{match.group(2)}") if match.group(2) else 0.0
    return warmup_epochs, min_factor


def parse_delayed_drop(lr_schedule):
    """(delay_epochs, drop_factor) of a schedule named eX_dropYY, e.g.
    e20_drop95 -> (20, 0.95), e5_drop75 -> (5, 0.75), e10_drop5 -> (10, 0.5),
    e10_drop05 -> (10, 0.05). The digits after "drop" are the decimals of the
    factor, so the factor is always between 0 and 1. None if lr_schedule does
    not have this form."""
    match = DELAYED_DROP_PATTERN.fullmatch(lr_schedule)
    if match is None:
        return None
    delay_epochs = int(match.group(1))
    drop_factor = float(f"0.{match.group(2)}")
    if drop_factor == 0:
        raise ValueError(
            f"Invalid learning rate schedule {lr_schedule}: the drop factor is 0"
        )
    return delay_epochs, drop_factor


def get_lr_scheduler(lr_schedule, optimizer, n_epochs):
    if lr_schedule == "constant":
        return get_constant_schedule(optimizer)
    if lr_schedule == "linear":
        return get_linear_schedule_with_warmup(optimizer, total_iters=n_epochs)

    delayed_drop = parse_delayed_drop(lr_schedule)
    if delayed_drop is not None:
        return get_delayed_drop_schedule(optimizer, *delayed_drop)

    warmup_cosine = parse_warmup_cosine(lr_schedule)
    if warmup_cosine is not None:
        return get_warmup_cosine_schedule(optimizer, *warmup_cosine, n_epochs)

    raise ValueError(
        f"Invalid learning rate schedule {lr_schedule}: use constant, linear, "
        "eX_dropYY (constant for X epochs, then multiplied by 0.YY every "
        "epoch, e.g. e20_drop95) or wX_cosineYY (linear warm-up for X epochs, "
        "then cosine decay to 0.YY of the learning rate at the last epoch, "
        "e.g. w5_cosine01)"
    )
