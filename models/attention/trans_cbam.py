import torch
import torch.nn as nn
import torch.nn.functional as F

class TransCBA(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, num_heads=8, attn_ratio=0.5):
        super(TransCBA, self).__init__()
        
        self.cross_attention = CrossAttention(in_channels, num_heads, attn_ratio)
        self.cb_cross = CBACross(in_channels, reduction_ratio)
        
    def forward(self, rgb, chm):
        rgb_attn, chm_attn = self.cross_attention(rgb, chm)
        
        return self.cb_cross(rgb_attn, chm_attn)
        

class CrossAttention(nn.Module): 
    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        super(CrossAttention, self).__init__()
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2

        self.rgb_qkv = nn.Conv2d(dim, h, kernel_size=1)
        self.chm_qkv = nn.Conv2d(dim, h, kernel_size=1)

        self.rgb_pe = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.chm_pe = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

        self.rgb_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.chm_proj = nn.Conv2d(dim, dim, kernel_size=1)

        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, rgb, chm):
        B, C, H, W = rgb.shape
        N = H * W
        
        rgb_qkv = self.rgb_qkv(rgb)
        rgb_q, rgb_k, rgb_v = rgb_qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        chm_qkv = self.chm_qkv(chm)
        chm_q, chm_k, chm_v = chm_qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        rgb_attn_weight = F.softmax(rgb_q.transpose(-2, -1) @ chm_k, dim=-1) * self.scale
        chm_attn_weight = F.softmax(chm_q.transpose(-2, -1) @ rgb_k, dim=-1) * self.scale
        
        rgb_attended = (rgb_v @ rgb_attn_weight.transpose(-2, -1)).view(B, C, H, W) + self.rgb_pe(rgb_v.reshape(B, C, H, W))
        chm_attended = (chm_v @ chm_attn_weight.transpose(-2, -1)).view(B, C, H, W) + self.chm_pe(chm_v.reshape(B, C, H, W))

        rgb_out = self.rgb_proj(rgb_attended)
        chm_out = self.chm_proj(chm_attended)

        return rgb_out, chm_out

class CBACross(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(CBACross, self).__init__()
    
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio

        self.rgb_channel_attn = ChannelAttention(in_channels, reduction_ratio)
        self.rgb_spatial_attn = SpatialAttention()
        self.chm_channel_attn = ChannelAttention(in_channels, reduction_ratio)
        self.chm_spatial_attn = SpatialAttention()

    def forward(self, rgb, chm): 
        rgb_sa = self.rgb_spatial_attn(rgb)
        rgb_ca = self.rgb_channel_attn(rgb)
        
        chm_sa = self.chm_spatial_attn(chm)
        chm_ca = self.chm_channel_attn(chm)
        
        attn_rgb = (rgb * chm_sa) * chm_ca
        attn_chm = (chm * rgb_sa) * rgb_ca

        return attn_rgb + attn_chm

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) 
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1))
        out = avg_out + max_out
        return self.sigmoid(out).view(x.size(0), x.size(1), 1, 1)

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv(out))
        return out