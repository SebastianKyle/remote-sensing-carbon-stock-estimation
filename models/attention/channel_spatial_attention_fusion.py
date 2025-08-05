import torch
import torch.nn as nn
import torch.nn.functional as F
from .hcf import ChannelAttention, SpatialAttention

class CSAF(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(CSAF, self).__init__()
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio

        self.rgb_channel_max = nn.AdaptiveMaxPool2d(1)
        self.rgb_channel_avg = nn.AdaptiveAvgPool2d(1)
        self.chm_channel_max = nn.AdaptiveMaxPool2d(1)
        self.chm_channel_avg = nn.AdaptiveAvgPool2d(1)
        self.rgb_ca_sigmoid = nn.Sigmoid()
        self.chm_ca_sigmoid = nn.Sigmoid()
        
        # self.rgb_spatial_max = nn.AdaptiveMaxPool2d((1, 1))
        # self.rgb_spatial_avg = nn.AdaptiveAvgPool2d((1, 1))
        # self.chm_spatial_max = nn.AdaptiveMaxPool2d((1, 1))
        # self.chm_spatial_avg = nn.AdaptiveAvgPool2d((1, 1))
        self.rgb_sa_sigmoid = nn.Sigmoid()
        self.chm_sa_sigmoid = nn.Sigmoid()

        reduced_channels = in_channels // reduction_ratio
        self.rgb_shared_mlp = nn.Sequential(
            nn.Linear(2 * in_channels, reduced_channels),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, reduced_channels),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, in_channels),
        )
        self.chm_shared_mlp = nn.Sequential(
            nn.Linear(2 * in_channels, reduced_channels),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, reduced_channels),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, in_channels),
        )

        self.rgb_conv = nn.Conv2d(6, 1, kernel_size=1)
        self.chm_conv = nn.Conv2d(6, 1, kernel_size=1)

    def forward(self, rgb, chm): 

        rgb_ca_max = self.rgb_channel_max(rgb)
        rgb_ca_avg = self.rgb_channel_avg(rgb)
        chm_ca_max = self.chm_channel_max(chm)
        chm_ca_avg = self.chm_channel_avg(chm)

        rgb_sa_max, _ = torch.max(rgb, dim=1, keepdim=True)
        rgb_sa_avg = torch.mean(rgb, dim=1, keepdim=True)
        chm_sa_max, _ = torch.max(chm, dim=1, keepdim=True)
        chm_sa_avg = torch.mean(chm, dim=1, keepdim=True)

        # Channel fusion
        ca_max_concat = torch.concat((rgb_ca_max, chm_ca_max), dim=1)
        ca_avg_concat = torch.concat((rgb_ca_avg, chm_ca_avg), dim=1)

        rgb_mlp_ca_max = self.rgb_shared_mlp(ca_max_concat.view(ca_max_concat.size(0), -1))
        rgb_mlp_ca_avg = self.rgb_shared_mlp(ca_avg_concat.view(ca_avg_concat.size(0), -1))
        rgb_ca = self.rgb_ca_sigmoid(rgb_mlp_ca_max + rgb_mlp_ca_avg).view(rgb.size(0), rgb.size(1), 1, 1)

        chm_mlp_ca_max = self.chm_shared_mlp(ca_max_concat.view(ca_max_concat.size(0), -1))
        chm_mlp_ca_avg = self.chm_shared_mlp(ca_avg_concat.view(ca_avg_concat.size(0), -1))
        chm_ca = self.chm_ca_sigmoid(chm_mlp_ca_max + chm_mlp_ca_avg).view(chm.size(0), chm.size(1), 1, 1)

        rgb_ca_attn = rgb * rgb_ca
        chm_ca_attn = chm * chm_ca

        # Spatial fusion
        sa_max_sum = rgb_sa_max + chm_sa_max
        sa_avg_sum = rgb_sa_avg + chm_sa_avg 

        sa_concat = torch.concat((rgb_sa_max, rgb_sa_avg, sa_max_sum, chm_sa_max, chm_sa_avg, sa_avg_sum), dim=1)
        rgb_sa = self.rgb_sa_sigmoid(self.rgb_conv(sa_concat))
        chm_sa = self.chm_sa_sigmoid(self.chm_conv(sa_concat))
        
        rgb_sa_attn = rgb * rgb_sa
        chm_sa_attn = chm * chm_sa

        return rgb + chm_ca_attn + chm_sa_attn, chm + rgb_ca_attn + rgb_sa_attn
        
