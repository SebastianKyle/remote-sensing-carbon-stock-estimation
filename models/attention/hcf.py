# Height-weighted Cross-Attention Fusion Module

import torch
import torch.nn as nn
import torch.nn.functional as F
from .cross_attention_fusion import CrossAttention
from einops import rearrange

class HCF(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, num_heads=8):
        super(HCF, self).__init__()
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio

        self.rgb_channel_attn = ChannelAttention(in_channels, reduction_ratio)
        self.rgb_spatial_attn = SpatialAttention()
        self.chm_channel_attn = ChannelAttention(in_channels, reduction_ratio)
        self.chm_spatial_attn = SpatialAttention()
        self.ca_fusion = CAWF(in_channels)

    def forward(self, rgb, chm):
        rgb_sa = self.rgb_spatial_attn(rgb)
        rgb_ca = self.rgb_channel_attn(rgb)
        
        chm_sa = self.chm_spatial_attn(chm)
        chm_ca = self.chm_channel_attn(chm)
        
        attn_rgb = (rgb * chm_sa) * chm_ca
        attn_chm = (chm * rgb_sa) * rgb_ca

        weight = self.ca_fusion(rgb, chm)

        return attn_rgb + attn_chm * weight

        # return attn_rgb + attn_chm

class HCF_RF(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(HCF_RF, self).__init__()
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio

        self.rgb_channel_attn = ChannelAttention(in_channels, reduction_ratio)
        self.rgb_spatial_attn = RFSpatialAttention(in_channels)
        self.chm_channel_attn = ChannelAttention(in_channels, reduction_ratio)
        self.chm_spatial_attn = RFSpatialAttention(in_channels)

        self.rgb_spatial_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=3),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True) 
        )

        self.chm_spatial_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=3),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True) 
        )

        self.ca_fusion = CAWF(in_channels)

    def forward(self, rgb, chm):
        
        rgb_sa, rgb_gen_feat = self.rgb_spatial_attn(rgb)
        rgb_ca = self.rgb_channel_attn(rgb)
        
        chm_sa, chm_gen_feat = self.chm_spatial_attn(chm)
        chm_ca = self.chm_channel_attn(chm)

        attn_rgb = self.rgb_spatial_conv((rgb_gen_feat * chm_ca) * chm_sa)
        attn_chm = self.chm_spatial_conv((chm_gen_feat * rgb_ca) * rgb_sa)
            
        weight = self.ca_fusion(rgb, chm)

        return attn_rgb + attn_chm * weight

class HCF_CA(nn.Module):
    
    def __init__(self, in_channels, reduction_ratio=16):
        super(HCF_CA, self).__init__()
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio

        self.rgb_channel_attn = ChannelAttention(in_channels, reduction_ratio)
        self.rgb_spatial_attn = SpatialAttention()
        self.chm_channel_attn = ChannelAttention(in_channels, reduction_ratio)
        self.chm_spatial_attn = SpatialAttention()
        # self.weighted_fusion = WeightedFusion(in_channels, 1)
        self.cross_attention_fusion = CrossAttention(in_channels, 8, 0.5)

    def forward(self, rgb, chm):
        rgb_sa = self.rgb_spatial_attn(rgb)
        rgb_ca = self.rgb_channel_attn(rgb)
        
        chm_sa = self.chm_spatial_attn(chm)
        chm_ca = self.chm_channel_attn(chm)
        
        attended_rgb = (rgb * chm_sa) * chm_ca
        attended_chm = (chm * rgb_sa) * rgb_ca

        return self.cross_attention_fusion(attended_rgb, attended_chm)


class HCF_LKA(nn.Module):
    def __init__(self, in_channels):
        super(HCF_LKA, self).__init__()
        self.in_channels = in_channels
        
        self.rgb_lka = nn.Sequential(    
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=5,
                padding=2,
                groups=in_channels
            ),    
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=7,
                padding=9,
                groups=in_channels,
                dilation=3
            ),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=1
            )
        )

        self.chm_lka = nn.Sequential(    
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=5,
                padding=2,
                groups=in_channels
            ),    
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=7,
                padding=9,
                groups=in_channels,
                dilation=3
            ),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=1
            )
        )
        
        self.weighted_fusion = WeightedFusion(in_channels, 1)
    
    def forward(self, rgb, chm):
        rgb_attn = self.rgb_lka(rgb)
        chm_attn = self.chm_lka(chm)
        
        attn_rgb = rgb * chm_attn
        attn_chm = chm * rgb_attn

        weight = self.weighted_fusion(rgb, chm)

        return attn_rgb + attn_chm * weight


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, separate=False):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) 
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            # nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            # nn.ReLU(),
            # nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)

            # nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            # nn.ReLU(),
            # nn.Linear(in_channels // reduction_ratio, in_channels, bias=False)

            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        self.sigmoid = nn.Sigmoid() if not separate else None
        self.sigmoid_m = nn.Sigmoid() if separate else None
        self.sigmoid_a = nn.Sigmoid() if separate else None
        self.separate = separate

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1))

        if self.separate == False:
            out = avg_out + max_out
            # return self.sigmoid(out)
            return self.sigmoid(out).view(x.size(0), x.size(1), 1, 1)

        return self.sigmoid_a(avg_out), self.sigmoid_m(max_out)

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        # self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

        # Gate conv
        # self.gate_conv = nn.Sequential(
        #     nn.Conv2d(2, 1, kernel_size=1),
        #     nn.Sigmoid()
        # )

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv(out))

        return out

        # gate = self.gate_conv(out)
        # out = gate * avg_out + (1 - gate) * max_out

        # return self.sigmoid(out)

class EfficientMultiScaleSpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Depthwise separable convolutions for different scales
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 2, kernel_size=3, padding=1, groups=2),  # depthwise
            nn.Conv2d(2, 1, kernel_size=1)  # pointwise
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(2, 2, kernel_size=3, padding=2, dilation=2, groups=2),  # depthwise
            nn.Conv2d(2, 1, kernel_size=1)  # pointwise
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(2, 2, kernel_size=3, padding=3, dilation=3, groups=2),  # depthwise
            nn.Conv2d(2, 1, kernel_size=1)  # pointwise
        )

        # Lightweight feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(3, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(2, 1, kernel_size=1)
        )
        
        # Gate mechanism for adaptive feature selection
        self.gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Get avg and max features
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate avg and max features
        feat = torch.cat([avg_out, max_out], dim=1)
        
        # Multi-scale feature extraction
        out1 = self.conv1(feat)
        out2 = self.conv2(feat)
        out3 = self.conv3(feat)
        
        # Concatenate multi-scale features
        multi_scale_feat = torch.cat([out1, out2, out3], dim=1)
        
        # Feature fusion
        fused_feat = self.fusion(multi_scale_feat)
        
        # Adaptive feature selection
        gate = self.gate(feat)
        
        # Final output with residual connection
        out = gate * avg_out + (1 - gate) * max_out + fused_feat
        
        return self.sigmoid(out)

class DilatedSpatialAttention(nn.Module):
    def __init__(self):
        super(DilatedSpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size=5, padding=2, dilation=1)
        self.conv2 = nn.Conv2d(2, 1, kernel_size=5, padding=4, dilation=2)
        # self.conv3 = nn.Conv2d(2, 1, kernel_size=5, padding=10, dilation=5)
        # self.conv4 = nn.Conv2d(2, 1, kernel_size=5, padding=2, dilation=1)
        # self.conv5 = nn.Conv2d(2, 1, kernel_size=5, padding=6, dilation=3)
        # self.conv6 = nn.Conv2d(2, 1, kernel_size=7, padding=3, dilation=1)

        self.proj = nn.Conv2d(2, 1, kernel_size=1)
        self.gate_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        out1 = self.conv1(torch.cat([avg_out, max_out], dim=1))
        out2 = self.conv2(torch.cat([avg_out, max_out], dim=1))

        # out = torch.cat([out1, out2], dim=1)
        # out = self.proj(out)

        gate = self.gate_conv(torch.cat([out1, out2], dim=1))  # [B,1,H,W]

        out = gate * out1 + (1 - gate) * out2  # Weighted sum

        return torch.sigmoid(out)

class RFSpatialAttention(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=1):
        super(RFSpatialAttention, self).__init__()
        self.kernel_size = kernel_size
        self.generate = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * (self.kernel_size ** 2), kernel_size=kernel_size, padding=kernel_size // 2, stride=stride, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels * (self.kernel_size ** 2)),
            nn.ReLU(inplace=True),
        )

        self.get_weight = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, h, w = x.size()

        generate_feature = self.generate(x).view(b, c, self.kernel_size ** 2, h, w)
        generate_feature = rearrange(generate_feature, 'b c (n1 n2) h w -> b c (h n1) (w n2)', n1=self.kernel_size, n2=self.kernel_size)

        max_feat, _ = torch.max(generate_feature, dim=1, keepdim=True)
        mean_feat = torch.mean(generate_feature, dim=1, keepdim=True)
        receptive_field_attn = self.get_weight(torch.cat([max_feat, mean_feat], dim=1))
        
        return receptive_field_attn, generate_feature

class WeightedFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(WeightedFusion, self).__init__()
        self.rgb_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.chm_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax1 = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.mlp = nn.Sequential(
        #     nn.Linear(in_channels, in_channels // 16),
        #     nn.ReLU(),
        #     nn.Linear(in_channels // 16, in_channels)
        # )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(ConvBNReLU(in_channels, 24, 1, 1), ConvBNSig(24, out_channels, 1, 1))
        
    def forward(self, rgb, chm):
        batch_size, c, h, w = rgb.size()
        proj_rgb = self.rgb_conv(rgb).view(batch_size, -1, h * w).permute(0, 2, 1) # (B, N, C)
        proj_chm = self.chm_conv(chm).view(batch_size, -1, h * w)                  # (B, C, N)
        
        energy = torch.bmm(proj_rgb, proj_chm)
        attn1 = self.softmax1(energy)
        
        attn_r = torch.bmm(proj_rgb.permute(0, 2, 1), attn1)
        attn_h = torch.bmm(proj_chm, attn1) 

        attn2 = attn_r + attn_h
        output = attn2.view(batch_size, c, h, w) + rgb + chm

        gate = self.mlp(self.gap(output))

        return gate
        
        
class ConvBNReLU(nn.Module):
    """Conv-BN-ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        if x.size(2) > 1 and x.size(3) > 1:  # Skip BN if spatial dimensions are 1x1
            x = self.bn(x)
        return self.relu(x)

class ConvBNSig(nn.Module):
    """Conv-BN-Sigmoid"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, **kwargs):
        super(ConvBNSig, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        if x.size(2) > 1 and x.size(3) > 1:  # Skip BN if spatial dimensions are 1x1
            x = self.bn(x)
        return self.sigmoid(x)

class CAWeightedFusion(nn.Module): 
    def __init__(self, dim, num_heads=8, attn_ratio=0.5, out_channels=1):
        super(CAWeightedFusion, self).__init__()
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2

        self.rgb_qkv = nn.Conv2d(dim, h, kernel_size=1)
        self.chm_qkv = nn.Conv2d(dim, h, kernel_size=1)
 
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim // 16),
            nn.ReLU(),
            nn.Linear(dim // 16, dim)
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(ConvBNReLU(dim, 24, 1, 1), ConvBNSig(24, out_channels, 1, 1))

        self.scale = self.key_dim ** -0.5

        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, rgb, chm):
        """
        """

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

        rgb_attn_weight = F.softmax((rgb_k @ chm_v.transpose(-2, -1)) * self.scale, dim=-1) # (B, key_dim, C)
        chm_attn_weight = F.softmax((chm_k @ rgb_v.transpose(-2, -1)) * self.scale, dim=-1) # (B, key_dim, C)

        rgb_attended = F.softmax(rgb_attn_weight.transpose(-2, -1) @ rgb_q, dim=-1).view(B, C, H, W)
        chm_attended = F.softmax(chm_attn_weight.transpose(-2, -1) @ chm_q, dim=-1).view(B, C, H, W)

        attn = rgb_attended + chm_attended

        output = attn + rgb + chm
        gate = self.mlp(self.gap(output))
        
        return gate 

class CAWF(nn.Module): 
    def __init__(self, channels, num_heads=8):
        super(CAWF, self).__init__()
        
        self.channels = channels
        self.reduced_channels = channels // 8
        
        # Query, Key, Value projections for RGB
        self.rgb_q = nn.Conv2d(channels, self.reduced_channels, kernel_size=1)
        self.rgb_k = nn.Conv2d(channels, self.reduced_channels, kernel_size=1)
        self.rgb_v = nn.Conv2d(channels, channels, kernel_size=1)
        
        # Query, Key, Value projections for depth
        self.depth_q = nn.Conv2d(channels, self.reduced_channels, kernel_size=1)
        self.depth_k = nn.Conv2d(channels, self.reduced_channels, kernel_size=1)
        self.depth_v = nn.Conv2d(channels, channels, kernel_size=1)

        # self.rgb_pe = nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels)
        # self.chm_pe = nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels)

        # self.rgb_proj = nn.Conv2d(channels, channels, kernel_size=1)
        # self.chm_proj = nn.Conv2d(channels, channels, kernel_size=1)

        self.scale = self.reduced_channels ** -0.5
        
        # Gated fusion layers
        self.gate_conv = nn.Sequential(
            nn.Conv2d(2*channels, channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            ConvBNReLU(channels, 24, 1, 1), 
            ConvBNSig(24, 1, 1, 1),
            nn.Sigmoid()
        ) 
        
    def forward(self, rgb, chm):
        B, C, H, W = rgb.shape

        # Cross-attention stage
        # Get Q,K,V for both modalities
        rgb_q = self.rgb_q(rgb).flatten(2)  # [N, C/8, H*W]
        rgb_k = self.rgb_k(rgb).flatten(2)  # [N, C/8, H*W]
        rgb_v = self.rgb_v(rgb).flatten(2)  # [N, C, H*W]
        
        chm_q = self.depth_q(chm).flatten(2)  # [N, C/8, H*W]
        chm_k = self.depth_k(chm).flatten(2)  # [N, C/8, H*W]
        chm_v = self.depth_v(chm).flatten(2)  # [N, C, H*W]

        rgb_attn = torch.softmax((rgb_q.transpose(-2, -1) @ chm_k) * self.scale, dim=-1) # [N, H*W, H*W]
        chm_attn = torch.softmax((chm_q.transpose(-2, -1) @ rgb_k) * self.scale, dim=-1) # [N, H*W, H*W]
        # rgb_attn = torch.softmax((rgb_q.transpose(-2, -1) @ chm_k), dim=-1) # [N, H*W, H*W]
        # chm_attn = torch.softmax((chm_q.transpose(-2, -1) @ rgb_k), dim=-1) # [N, H*W, H*W]
        
        rgb_attended = (rgb_v @ rgb_attn).view_as(rgb) # [N, C, H, W]
        chm_attended = (chm_v @ chm_attn).view_as(chm) # [N, C, H, W] 

        # rgb_pp = self.rgb_pe(rgb_attended)
        # chm_pp = self.chm_pe(chm_attended)

        # rgb_attended = self.rgb_proj(rgb_attended + rgb_pp)
        # chm_attended = self.chm_proj(chm_attended + chm_pp)

        c = torch.cat([rgb_attended, chm_attended], dim=1)  # [N, 2C, H, W]
        
        g = self.gate_conv(c) # [N, C, H, W]

        c_fused = rgb_attended * g + chm_attended * (1 - g)  # [N, C, H, W]
        d_fused = rgb_attended * chm_attended  # [N, C, H, W]
        output = c_fused + d_fused

        # output = rgb_attended * g + chm_attended * (1 - g) # [N, C, H, W] 

        gate = self.mlp(output)

        return gate