import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ConstantLR, LinearLR


def get_constant_schedule(optimizer: Optimizer):
    return ConstantLR(optimizer, factor=1.0)


def get_linear_schedule_with_warmup(optimizer: Optimizer, total_iters: int):
    return LinearLR(
        optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_iters
    )

def get_lr_scheduler(lr_schedule, optimizer, n_epochs):
    if lr_schedule == "constant":
        scheduler = get_constant_schedule(optimizer)
    elif lr_schedule == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer, total_iters=n_epochs
        )
    else:
        raise ValueError("Invalid learning rate schedule")

    return scheduler
