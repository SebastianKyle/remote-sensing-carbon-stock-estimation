import torch
from torch import nn

class Concat(nn.Module):
    """Concatenate a list of tensors along dimension"""
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)