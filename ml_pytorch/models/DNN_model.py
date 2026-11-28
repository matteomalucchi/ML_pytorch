from torch import nn
import torch

from ml_pytorch.utils.learning_rate_schedules import get_lr_scheduler
from ml_pytorch.utils.optimizers import get_optimizer


class DNN(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(dim_in, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        logits = self.sigmoid(logits)
        return logits


def get_model(input_size, device, lr, lr_schedule, n_epochs, optimizer="adam", weight_decay=0.0):
    model = DNN(input_size).to(device)
    print(model)

    loss_fn = torch.nn.BCELoss(reduction="none")
    # optimizer and weight decay from the config (optimizer, weight_decay)
    optimizer = get_optimizer(optimizer, model.parameters(), lr, weight_decay)
    scheduler = get_lr_scheduler(lr_schedule, optimizer, n_epochs)

    return model, loss_fn, optimizer, scheduler
