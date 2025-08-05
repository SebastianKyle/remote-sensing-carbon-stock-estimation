import torch
import torch.nn as nn
import torch.nn.functional as F

class CAF(nn.Module):
    def __init__(self, in_channels, num_heads=8):
        super(CAF, self).__init__()
        
        self.in_channels = in_channels
        self.reduced_channels = in_channels // 8
        self.num_heads = num_heads
        self.head_dim = self.reduced_channels // num_heads
        
        self.rgb_query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.rgb_key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.rgb_value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.chm_query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.chm_key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.chm_value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.rgb_pe = nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3, groups=in_channels)
        self.chm_pe = nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3, groups=in_channels)

        self.rgb_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.chm_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.rgb_softmax = nn.Softmax(dim=-1)
        self.chm_softmax = nn.Softmax(dim=-1)

        self.pos_enc = nn.Parameter(torch.randn(1, self.reduced_channels, 32, 32))

        self.scale = self.reduced_channels ** -0.5

        # self.res_main = nn.Sequential(
        #     nn.Conv2d(2 * in_channels, in_channels, kernel_size=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels, in_channels, kernel_size=1)
        # )
        # self.res_skip = nn.Conv2d(2 * in_channels, in_channels, kernel_size=1)
        # self.bn = nn.BatchNorm2d(in_channels)

        self.gate_conv = nn.Sequential(
            nn.Conv2d(2*in_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, rgb, chm):
        B, C, H, W = rgb.size()

        rgb_q = self.rgb_query_conv(rgb).view(B, -1, H * W)
        rgb_k = self.rgb_key_conv(rgb).view(B, -1, H * W)
        rgb_v = self.rgb_value_conv(rgb)

        chm_q = self.rgb_query_conv(chm).view(B, -1, H * W)
        chm_k = self.rgb_key_conv(chm).view(B, -1, H * W)
        chm_v = self.rgb_value_conv(chm)

        pos_enc = F.interpolate(self.pos_enc, size=(H, W), mode='bilinear', align_corners=False)

        rgb_q = rgb_q + pos_enc.view(B, -1, H * W)
        rgb_k = rgb_k + pos_enc.view(B, -1, H * W)
        chm_q = chm_q + pos_enc.view(B, -1, H * W)
        chm_k = chm_k + pos_enc.view(B, -1, H * W)

        rgb_attn_weight = self.rgb_softmax(torch.bmm(rgb_q.transpose(1, 2), chm_k) * self.scale)
        chm_attn_weight = self.chm_softmax(torch.bmm(chm_q.transpose(1, 2), rgb_k) * self.scale)
        
        rgb_attn = torch.bmm(rgb_v.view(B, -1, H * W), rgb_attn_weight).view(B, C, H, W)
        chm_attn = torch.bmm(chm_v.view(B, -1, H * W), chm_attn_weight).view(B, C, H, W)

        rgb_pp = self.rgb_pe(rgb_v)
        chm_pp = self.chm_pe(chm_v)
        
        rgb_attn = self.rgb_proj(rgb_attn + rgb_pp)
        chm_attn = self.chm_proj(chm_attn + chm_pp)
        
        attn_concat = torch.cat([rgb_attn, chm_attn], dim=1)
        gate = self.gate_conv(attn_concat)
        out = rgb_attn * gate + chm_attn * (1 - gate)
        # out = self.bn(self.res_main(attn_concat) + self.res_skip(attn_concat))
        
        return out, rgb_attn, chm_attn

class CrossAttention(nn.Module):
    
    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        super(CrossAttention, self).__init__()
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2

        self.rgb_qkv = nn.Conv2d(dim, h, kernel_size=1)
        self.chm_qkv = nn.Conv2d(dim, h, kernel_size=1)

        self.proj = nn.Conv2d(dim, dim, kernel_size=1)

        self.rgb_pe = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.chm_pe = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

        self.res_main = nn.Sequential(
            nn.Conv2d(2 * dim, dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=1)
        )
        self.res_skip = nn.Conv2d(2 * dim, dim, kernel_size=1)

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

        rgb_attn_weight = F.softmax(rgb_k @ rgb_v.transpose(-2, -1), dim=-1)
        chm_attn_weight = F.softmax(chm_k @ chm_v.transpose(-2, -1), dim=-1)
        
        rgb_attended = (chm_attn_weight.transpose(-2, -1) @ rgb_q).view(B, C, H, W) + self.rgb_pe(rgb_v.reshape(B, C, H, W))
        chm_attended = (rgb_attn_weight.transpose(-2, -1) @ chm_q).view(B, C, H, W) + self.chm_pe(chm_v.reshape(B, C, H, W))

        return rgb_attended + chm_attended

        # output = torch.cat([rgb_attended, chm_attended], dim=1)

        # return self.res_main(output) + self.res_skip(output)


        



        