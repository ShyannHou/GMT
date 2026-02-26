import torch
import torch.nn as nn
import torch.nn.functional as F


# DNN
class CNN(nn.Module):
    def __init__(self, input_channel, dim_out):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(input_channel, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # a → a/2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, dim_out)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class ExpandMLP(nn.Module):
    def __init__(self, d, n):
        super().__init__()
        self.n = n
        self.d = d

        self.mlp = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, n * d),
        )

    def forward(self, x):
        # x: (B, d)
        x = self.mlp(x)  # (B, n*d)
        x = x.view(x.size(0), self.n, self.d)  # (B, n, d)
        return x
