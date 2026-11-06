import re

from torch.optim import Optimizer
from torch.optim.lr_scheduler import ConstantLR, LinearLR, LambdaLR

# eX_dropYY: constant for X epochs, then multiplied by 0.YY every epoch
DELAYED_DROP_PATTERN = re.compile(r"e(\d+)_drop(\d+)")


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

    raise ValueError(
        f"Invalid learning rate schedule {lr_schedule}: use constant, linear "
        "or eX_dropYY (constant for X epochs, then multiplied by 0.YY every "
        "epoch, e.g. e20_drop95)"
    )
