import torch
import torch.nn as nn
import torch.nn.functional as F
from .gated import GatedFusion

class LKAFusion(nn.Module):
    def __init__(self, in_channels):
        super(LKAFusion, self).__init__()
        self.rgb_lka = LKA(in_channels)
        self.chm_lka = LKA(in_channels)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, rgb, chm):
        rgb = self.rgb_lka(rgb)
        chm = self.chm_lka(chm)

        return rgb + self.gamma * chm

class LKAFusion1(nn.Module):
    def __init__(self, in_channels):
        super(LKAFusion1, self).__init__()
        self.rgb_lka = LKA(in_channels)
        self.chm_lka = LKA(in_channels)

        self.fuse_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        self.gelu = nn.GELU()

    def forward(self, rgb, chm):
        rgb = self.rgb_lka(rgb)
        chm = self.chm_lka(chm)
        cat = torch.cat([rgb, chm], dim=1)

        return self.fuse_conv(self.gelu(cat))

class LKA(nn.Module):
    def __init__(self, channels):
        super(LKA, self).__init__()
        
        self.dwconv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=5,
            padding=2,
            groups=channels
        )

        self.dwdconv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=7,
            padding=9,
            groups=channels,
            dilation=3
        )

        self.pwconv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1
        )

        self.channel_conv1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.channel_conv2 = nn.Conv2d(channels, channels, kernel_size=1)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.channel_conv1(x)
        x = self.gelu(x)

        weight = self.pwconv(self.dwdconv(self.dwconv(x)))

        x = x * weight
        x = self.channel_conv2(x)

        return x

    # def forward(self, rgb_features, chm_features):
    #     # x = rgb_features + chm_features * self.gamma
    #     # x = torch.concat((rgb_features, chm_features), dim=1)
    #     x = self.gated(rgb_features, chm_features)
        
    #     x = self.channel_conv1(x)
    #     x = self.gelu(x)

    #     weight = self.pwconv(self.dwdconv(self.dwconv(x)))

    #     x = x * weight
    #     x = self.channel_conv2(x)

    #     return x