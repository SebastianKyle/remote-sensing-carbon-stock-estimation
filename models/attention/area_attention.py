import torch
import torch.nn as nn
import torch.nn.functional as F
from models.yolo.conv import Conv

class AAttnFusion(nn.Module):
    """
    Fusion module which fuses rgb and chm feature maps using area-attention mechanism.
    

    """
    def __init__(self, in_channels, area=1):
        """Initializes the fusion module."""
        super().__init__()
        self.area = area

        self.rgb_conv1 = Conv(in_channels, in_channels // 2, 1)
        self.chm_conv1 = Conv(in_channels, in_channels // 2, 1)
        self.fuse_conv = Conv(in_channels, in_channels, 1)
        self.gelu = nn.GELU()
        
        dim = in_channels // 2
        num_heads = dim // 32
        self.rgb_aattn = AAttn(dim, num_heads, area)
        self.chm_aattn = AAttn(dim, num_heads, area)
        
    def forward(self, rgb, chm):
        rgb = self.rgb_aattn(self.rgb_conv1(rgb))
        chm = self.chm_aattn(self.chm_conv1(chm))
        cat = torch.cat([rgb, chm], dim=1)
        
        return self.fuse_conv(self.gelu(cat))

class AAttn(nn.Module):
    """
    Area-attention module with the requirement of flash attention.

    Attributes:
        dim (int): Number of hidden channels;
        num_heads (int): Number of heads into which the attention mechanism is divided;
        area (int, optional): Number of areas the feature map is divided. Defaults to 1.

    Methods:
        forward: Performs a forward process of input tensor and outputs a tensor after the execution of the area attention mechanism.

    Examples:
        >>> import torch
        >>> from ultralytics.nn.modules import AAttn
        >>> model = AAttn(dim=64, num_heads=2, area=4)
        >>> x = torch.randn(2, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    
    Notes: 
        recommend that dim//num_heads be a multiple of 32 or 64.

    """

    def __init__(self, dim, num_heads, area=1):
        """Initializes the area-attention module, a simple yet efficient attention module for YOLO."""
        super().__init__()
        self.area = area

        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        all_head_dim = head_dim * self.num_heads

        self.qkv = Conv(dim, all_head_dim * 3, 1, act=False)
        self.proj = Conv(all_head_dim, dim, 1, act=False)

        self.pe = Conv(all_head_dim, dim, 7, 1, 3, g=dim, act=False, bias=True)


    def forward(self, x):
        """Processes the input tensor 'x' through the area-attention"""
        B, C, H, W = x.shape
        N = H * W

        qkv = self.qkv(x)
        qk, v = qkv.split([C * 2, C], dim=1)

        pp = self.pe(v)
        v = v.flatten(2)
        if self.area > 1:
            qk = qk.reshape(B * self.area, C * 2, N // self.area)
            v = v.reshape(B * self.area, C, N // self.area)
            B, _, N = qk.shape

        q, k = qk.split([C, C], dim=1)
        q = q.view(B, self.num_heads, self.head_dim, N)
        k = k.view(B, self.num_heads, self.head_dim, N)
        v = v.view(B, self.num_heads, self.head_dim, N)
        attn = (q.transpose(-2, -1) @ k) * (self.head_dim ** -0.5)
        max_attn = attn.max(dim=-1, keepdim=True).values
        exp_attn = torch.exp(attn - max_attn)
        attn = exp_attn / exp_attn.sum(dim=-1, keepdim=True)
        x = (v @ attn.transpose(-2, -1))

        if self.area > 1:
            x = x.reshape(B // self.area, C, N * self.area)
            B, _, N = x.shape
        x = x.reshape(B, C, H, W)

        return self.proj(x + pp)