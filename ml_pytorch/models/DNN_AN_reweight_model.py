from torch import nn
import torch

from ml_pytorch.utils.learning_rate_schedules import get_lr_scheduler


class DNN(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.BatchNorm1d(dim_in),
            nn.Linear(dim_in, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        #logits = self.softmax(logits)
        return logits


def get_model(input_size, device, lr, lr_schedule, n_epochs):
    model = DNN(input_size).to(device)
    print(model)

    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = get_lr_scheduler(lr_schedule, optimizer, n_epochs)

    return model, loss_fn, optimizer, scheduler
