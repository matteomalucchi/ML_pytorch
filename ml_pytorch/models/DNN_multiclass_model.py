from torch import nn
import torch

from ml_pytorch.utils.learning_rate_schedules import get_lr_scheduler


class MaskedBatchNorm1d(nn.Module):
    """BatchNorm1d on raw inputs that ignores padded entries.

    Entries equal to ``pad_value`` (e.g. the -999 PocketCoffea uses for missing
    jets) are excluded from the batch and running statistics, so the padding
    does not dominate the mean/variance. After the normalisation the padded
    entries are replaced by ``pad_fill_value`` (in normalised units, 0 = mean of
    the real values) or, with ``pad_fill_value="learn"``, by a learnable value
    per feature. With ``pad_indicators`` one 0/1 flag per feature (1 = real
    value) is appended, so the output has ``2 * num_features`` columns.

    Everything lives inside the model, so the exported ONNX takes the raw,
    padded inputs directly.
    """

    def __init__(
        self,
        num_features: int,
        pad_value: float = -999.0,
        pad_fill_value=0.0,
        pad_indicators: bool = True,
        eps: float = 1e-5,
        momentum: float = 0.1,
    ):
        super().__init__()
        self.pad_value = float(pad_value)
        self.pad_indicators = pad_indicators
        self.eps = eps
        self.momentum = momentum

        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

        if pad_fill_value == "learn":
            self.pad_fill = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_buffer(
                "pad_fill", torch.full((num_features,), float(pad_fill_value))
            )

        self.num_features_out = num_features * (2 if pad_indicators else 1)

    def forward(self, x):
        is_real = x != self.pad_value

        if self.training:
            w = is_real.to(x.dtype)
            n = w.sum(dim=0)
            n_safe = n.clamp(min=1.0)
            batch_mean = (x * w).sum(dim=0) / n_safe
            batch_var = (((x - batch_mean) ** 2) * w).sum(dim=0) / n_safe
            # features without (enough) real entries in this batch fall back
            # to the running statistics instead of producing nan
            mean = torch.where(n > 0, batch_mean, self.running_mean)
            var = torch.where(n > 1, batch_var, self.running_var)
            with torch.no_grad():
                unbiased_var = batch_var * n / (n - 1).clamp(min=1.0)
                self.running_mean.copy_(
                    torch.where(
                        n > 0,
                        (1 - self.momentum) * self.running_mean + self.momentum * batch_mean,
                        self.running_mean,
                    )
                )
                self.running_var.copy_(
                    torch.where(
                        n > 1,
                        (1 - self.momentum) * self.running_var + self.momentum * unbiased_var,
                        self.running_var,
                    )
                )
        else:
            mean = self.running_mean
            var = self.running_var

        x_norm = (x - mean) / torch.sqrt(var + self.eps) * self.weight + self.bias
        x_norm = torch.where(is_real, x_norm, self.pad_fill.expand_as(x_norm))

        if self.pad_indicators:
            x_norm = torch.cat([x_norm, is_real.to(x_norm.dtype)], dim=1)
        return x_norm


class DNN(nn.Module):
    def __init__(
        self,
        dim_in: int,
        num_classes: int,
        pad_value: float = -999.0,
        pad_fill_value=0.0,
        pad_indicators: bool = True,
    ):
        super().__init__()

        self.input_norm = MaskedBatchNorm1d(
            dim_in,
            pad_value=pad_value,
            pad_fill_value=pad_fill_value,
            pad_indicators=pad_indicators,
        )
        self.net = nn.Sequential(
            nn.Linear(self.input_norm.num_features_out, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.input_norm(x)
        return self.net(x)

    def export_model(self, model):
        class ONNXWrappedModel(torch.nn.Module):
            def __init__(self, original_model):
                super().__init__()
                self.model = original_model

            def forward(self, x):
                logits = self.model(x)
                return torch.nn.functional.softmax(logits, dim=1)

        return ONNXWrappedModel(model)


def get_model(
    input_size,
    num_classes,
    device,
    lr,
    lr_schedule,
    n_epochs,
    pad_value=-999.0,
    pad_fill_value=0.0,
    pad_indicators=True,
):
    model = DNN(
        input_size,
        num_classes,
        pad_value=pad_value,
        pad_fill_value=pad_fill_value,
        pad_indicators=pad_indicators,
    ).to(device)
    print(model)

    # Multi-class single-label classification:
    #   model outputs raw logits [N, C]
    #   targets are int64 class indices [N]
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = get_lr_scheduler(lr_schedule, optimizer, n_epochs)

    return model, loss_fn, optimizer, scheduler
