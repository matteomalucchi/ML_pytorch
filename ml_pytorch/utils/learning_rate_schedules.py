import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ConstantLR, LinearLR, LambdaLR


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
    
def get_lr_scheduler(lr_schedule, optimizer, n_epochs):
    if lr_schedule == "constant":
        scheduler = get_constant_schedule(optimizer)
    elif lr_schedule == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer, total_iters=n_epochs
        )
    elif lr_schedule == "e20_drop75":
        scheduler = get_delayed_drop_schedule(optimizer, 20, 0.75)
    elif lr_schedule == "e5_drop75":
        scheduler = get_delayed_drop_schedule(optimizer, 5, 0.75)
    elif lr_schedule == "e20_drop95":
        scheduler = get_delayed_drop_schedule(optimizer, 20, 0.95)
    elif lr_schedule == "e30_drop95":
        scheduler = get_delayed_drop_schedule(optimizer, 30, 0.95)
    else:
        raise ValueError("Invalid learning rate schedule")

    return scheduler
