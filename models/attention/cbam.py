import torch
import torch.nn as nn
import torch.nn.functional as F
from .gated import GatedFusion

class CBAMFusion(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(CBAMFusion, self).__init__()
        self.rgb_ca = CBAM(in_channels, reduction_ratio)
        self.chm_ca = CBAM(in_channels, reduction_ratio)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, rgb, chm):
        rgb = self.rgb_ca(rgb)
        chm = self.chm_ca(chm)

        return rgb + self.gamma * chm

class CBAMFusion1(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(CBAMFusion1, self).__init__()
        self.conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        self.rgb_ca = CBAM(in_channels, reduction_ratio)
        self.chm_ca = CBAM(in_channels, reduction_ratio)
        self.gelu = nn.GELU()

    def forward(self, rgb, chm):
        rgb = self.rgb_ca(rgb)
        chm = self.chm_ca(chm)
        cat = torch.cat([rgb, chm], dim=1)

        return self.conv(self.gelu(cat))

class CBAMFusion2(nn.Module): 
    def __init__(self, in_channels, reduction_ratio=16):
        super(CBAMFusion2, self).__init__()
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1) 
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        self.sigmoid = nn.Sigmoid()
        
        # Spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)

        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)
        
    def channel_attention(self, x):
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1))
        out = avg_out + max_out
        return self.sigmoid(out).view(x.size(0), x.size(1), 1, 1)

    def spatial_attention(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv(out))
        return out

    def forward(self, rgb, chm):
        ca = self.channel_attention(rgb)
        rgb = rgb * ca
        sa = self.spatial_attention(chm)
        rgb = rgb * sa

        return rgb

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1) 
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        self.sigmoid = nn.Sigmoid()
        
        # Spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)

        # self.channel_reduction = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

        self.gamma = nn.Parameter(torch.ones(1))

        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)
        
    def channel_attention(self, x):
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1))
        out = avg_out + max_out
        return self.sigmoid(out).view(x.size(0), x.size(1), 1, 1)

    def spatial_attention(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv(out))
        return out

    def forward(self, x):
        ca = self.channel_attention(x)
        x = x * ca
        sa = self.spatial_attention(x)
        x = x * sa

        return x

    # def forward(self, rgb_features, chm_features):
    #     add = rgb_features + chm_features * self.gamma
    #     ca = self.channel_attention(add)
    #     add = add * ca
        
    #     sa = self.spatial_attention(add)
    #     out = add * sa

    #     return out

        # rgb_ca = self.channel_attention(rgb_features)
        # rgb_features = rgb_features * rgb_ca
        # rgb_sa = self.spatial_attention(rgb_features)
        # rgb_features = rgb_features * rgb_sa

        # chm_ca = self.channel_attention(chm_features)
        # chm_features = chm_features * chm_ca
        # chm_sa = self.spatial_attention(chm_features)
        # chm_features = chm_features * chm_sa

        # return rgb_features, chm_features

        # combined_features = torch.cat([rgb_features, chm_features], dim=1)
        # ca = self.channel_attention(combined_features)
        # fused = combined_features * ca

        # sa = self.spatial_attention(fused)
        # fused = fused * sa
        
        # fused = self.channel_attention(fused)
        
        # return fused