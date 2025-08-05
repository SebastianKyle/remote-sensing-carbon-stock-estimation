import torch
import torch.nn as nn
import torch.nn.functional as F
from .gated import GatedFusion

class MSCAFusion(nn.Module):
    def __init__(self, in_channels):
        super(MSCAFusion, self).__init__()
        self.rgb_msca = MSCA(in_channels)
        self.chm_msca = MSCA(in_channels)

        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, rgb, chm):
        rgb = self.rgb_msca(rgb)
        chm = self.chm_msca(chm)

        return rgb + self.gamma * chm

class MSCAFusion1(nn.Module):
    def __init__(self, in_channels):
        super(MSCAFusion1, self).__init__()
        self.rgb_msca = MSCA(in_channels)
        self.chm_msca = MSCA(in_channels)

        self.fuse_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        self.gelu = nn.GELU()
    
    def forward(self, rgb, chm):
        rgb = self.rgb_msca(rgb)
        chm = self.chm_msca(chm)

        cat = torch.cat([rgb, chm], dim=1)

        return self.fuse_conv(self.gelu(cat))

class MSCA(nn.Module):
    def __init__(self,
                 channels):
        super(MSCA, self).__init__()
        self.dwconv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=5,
            padding=2,
            groups=channels
        )
        self.scale_7 = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=(1, 7),
                padding=(0, 3),
                groups=channels
            ),
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=(7, 1),
                padding=(3, 0),
                groups=channels
            )
        )
        self.scale_11 = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=(1, 11),
                padding=(0, 5),
                groups=channels
            ),
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=(11, 1),
                padding=(5, 0),
                groups=channels
            )
        )
        self.scale_21 = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=(1, 21),
                padding=(0, 10),
                groups=channels
            ),
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=(21, 1),
                padding=(10, 0),
                groups=channels
            )
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

        base_weight = self.dwconv(x)
        weight1 = self.scale_7(base_weight)
        weight2 = self.scale_11(base_weight)
        weight3 = self.scale_21(base_weight)
        weight = base_weight + weight1 + weight2 + weight3
        weight = self.pwconv(weight)

        x = x * weight
        x = self.channel_conv2(x)

        return x

    # def forward(self, rgb_features, chm_features):
    #     x = self.gated(rgb_features, chm_features)
    #     x = self.channel_conv1(x)
    #     x = self.gelu(x)

    #     base_weight = self.dwconv(x)
    #     weight1 = self.scale_7(base_weight)
    #     weight2 = self.scale_11(base_weight)
    #     weight3 = self.scale_21(base_weight)
    #     weight = base_weight + weight1 + weight2 + weight3
    #     weight = self.pwconv(weight)

    #     x = x * weight
    #     x = self.channel_conv2(x)

    #     return x