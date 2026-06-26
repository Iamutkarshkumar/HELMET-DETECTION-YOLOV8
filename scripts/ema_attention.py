# ema.py
import torch
import torch.nn as nn

class EMAAttention(nn.Module):
    """
    Exponential Moving Average Channel Attention
    """
    def __init__(self, channels, decay=0.95):
        super().__init__()
        self.decay = decay
        self.mu = nn.Parameter(
            torch.zeros(1, channels, 1, 1),
            requires_grad=False
        )
        self.gamma = nn.Parameter(
            torch.ones(1, channels, 1, 1)
        )

    def forward(self, x):
        with torch.no_grad():
            self.mu.mul_(self.decay).add_(
                x.mean(dim=(0, 2, 3), keepdim=True) * (1 - self.decay)
            )
        return x * self.gamma + self.mu
