import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ConstantLR, LinearLR, LambdaLR


def get_constant_schedule(optimizer: Optimizer):
    return ConstantLR(optimizer, factor=1.0)


def get_linear_schedule_with_warmup(optimizer: Optimizer, total_iters: int):
    return LinearLR(
        optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_iters
    )

def get_delayed_drop_schedule(optimizer: Optimizer):
    lambda1 = lambda epoch: (0.75**(epoch-5) if epoch>=6 else 1)
    return LambdaLR(
        optimizer, lr_lambda=lambda1
    )

def get_lr_scheduler(lr_schedule, optimizer, n_epochs):
    if lr_schedule == "constant":
        scheduler = get_constant_schedule(optimizer)
    elif lr_schedule == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer, total_iters=n_epochs
        )
    elif lr_schedule == "delayed_drop":
        scheduler = get_delayed_drop_schedule(optimizer)


    else:
        raise ValueError("Invalid learning rate schedule")

    return scheduler
