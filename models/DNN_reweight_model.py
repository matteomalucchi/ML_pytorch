from torch import nn
import torch
import sys
import os

sys.path.append(f"{os.path.dirname(__file__)}/../")
from scripts.learning_rate_schedules import get_lr_scheduler


class DNN(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(dim_in, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        logits = self.sigmoid(logits)
        return logits


def get_model(input_size, device, lr, lr_schedule, n_epochs):
    model = DNN(input_size).to(device)
    print(model)

    loss_fn = torch.nn.BCELoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = get_lr_scheduler(lr_schedule, optimizer, n_epochs)

    return model, loss_fn, optimizer, scheduler
