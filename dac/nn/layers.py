import cached_conv as cc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils import weight_norm


class WNConv1d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv = cc.Conv1d(*args, **kwargs)
        self.wn = weight_norm(self.conv)

    def forward(self, x):
        return self.conv(x)

    @property
    def cumulative_delay(self):
        return self.conv.cumulative_delay


class WNConvTranspose1d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv = cc.ConvTranspose1d(*args, **kwargs)
        self.wn = weight_norm(self.conv)

    def forward(self, x):
        return self.conv(x)

    @property
    def cumulative_delay(self):
        return self.conv.cumulative_delay


# Scripting this brings model speed up 1.4x
@torch.jit.script
def snake(x, alpha):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


class Snake1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        res = snake(x, self.alpha)
        return res
