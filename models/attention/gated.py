import torch
from torch import nn

class GatedFusion(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(2*in_channels, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, rgb_features, chm_features):
        combined = torch.cat([rgb_features, chm_features], dim=1)
        gate = self.gate(combined)
        return gate * rgb_features + (1 - gate) * chm_features