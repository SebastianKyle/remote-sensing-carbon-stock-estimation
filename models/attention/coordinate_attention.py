import torch
import torch.nn as nn
import torch.nn.functional as F

class CA_FIM(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        """
        Coordinate Attention Feature Interaction Module (CA-FIM)
        
        Args:
            channels: Number of input channels (same for RGB and depth features)
            reduction_ratio: Reduction ratio for channel reduction in MLP
        """
        super(CA_FIM, self).__init__()
        
        self.channels = channels
        self.reduction_ratio = reduction_ratio
        reduced_channels = channels // reduction_ratio
        
        self.mlp = nn.Sequential(
            nn.Linear(channels, reduced_channels),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, reduced_channels),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, reduced_channels),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, reduced_channels),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels)
        )
        
        # 1x1 convolutions for transformation
        self.conv_h_r = nn.Conv1d(channels, channels, kernel_size=1)
        self.conv_w_r = nn.Conv1d(channels, channels, kernel_size=1)
        self.conv_h_d = nn.Conv1d(channels, channels, kernel_size=1)
        self.conv_w_d = nn.Conv1d(channels, channels, kernel_size=1)
        
    def forward(self, rgb_feat, chm_feat):
        """
        Args:
            rgb_feat: RGB features [N, C, H, W]
            chm_feat: Depth features [N, C, H, W]
        Returns:
            Ar: Enhanced RGB features [N, C, H, W]
            Ad: Enhanced depth features [N, C, H, W]
        """
        N, C, H, W = rgb_feat.size()

        # Horizontal pooling (H,1) - for vertical positional encoding
        rgb_h = torch.mean(rgb_feat, dim=3, keepdim=True)  # [N,C,H,1]
        depth_h = torch.mean(chm_feat, dim=3, keepdim=True)
        
        # Vertical pooling (1,W) - for horizontal positional encoding
        rgb_w = torch.mean(rgb_feat, dim=2, keepdim=True)  # [N,C,1,W]
        depth_w = torch.mean(chm_feat, dim=2, keepdim=True)
        
        # Concatenate and MLP
        rgb_h = rgb_h.permute(0,2,1,3).flatten(2)  # [N,H,C]
        rgb_w = rgb_w.permute(0,3,1,2).flatten(2)  # [N,W,C]
        depth_h = depth_h.permute(0,2,1,3).flatten(2)  # [N,H,C]
        depth_w = depth_w.permute(0,3,1,2).flatten(2)  # [N,W,C]
        
        # Concatenate all features
        combined = torch.cat([rgb_h, rgb_w, depth_h, depth_w], dim=1)  # [N,2(H+W),C]
        
        # Apply MLP
        f = self.mlp(combined) # [N, 2(H+W), 4C]
        
        # Split into four parts
        f_h_r = f[:, :H, :]  # [N,H,C]
        f_w_r = f[:, H:(H+W), :]  # [N,W,C]
        f_h_d = f[:, (H+W):(2*H+W), :]  # [N,H,C]
        f_w_d = f[:, (2*H+W):, :]  # [N,W,C]
        
        # Reshape and permute for convolution
        f_h_r = f_h_r.permute(0,2,1)
        f_w_r = f_w_r.permute(0,2,1)
        f_h_d = f_h_d.permute(0,2,1)
        f_w_d = f_w_d.permute(0,2,1)
        
        # Apply 1D convolutions
        w_h_r = torch.sigmoid(self.conv_h_r(f_h_r))  # [N,C,H,1]
        w_w_r = torch.sigmoid(self.conv_w_r(f_w_r))  # [N,C,1,W]
        w_h_d = torch.sigmoid(self.conv_h_d(f_h_d))  # [N,C,H,1]
        w_w_d = torch.sigmoid(self.conv_w_d(f_w_d))  # [N,C,1,W]

        # Reshape back to 4D tensors for further processing
        w_h_r = w_h_r.unsqueeze(-1)  # [N, C, H, 1]
        w_w_r = w_w_r.unsqueeze(-2)  # [N, C, 1, W]
        w_h_d = w_h_d.unsqueeze(-1)  # [N, C, H, 1]
        w_w_d = w_w_d.unsqueeze(-2)  # [N, C, 1, W]
        
        # Apply attention weights
        # For RGB features
        ar = chm_feat * w_h_r * w_w_r + rgb_feat
        
        # For depth features
        ad = rgb_feat * w_h_d * w_w_d + chm_feat
        
        return ar, ad

class GC_FFM(nn.Module):
    def __init__(self, channels):
        """
        Gated Cross-Attention Feature Fusion Module (GC-FFM)
        
        Args:
            channels: Number of input channels (same for RGB and depth features)
        """
        super(GC_FFM, self).__init__()
        
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
        
        # Gated fusion layers
        self.gate_conv = nn.Sequential(
            nn.Conv2d(2*channels, channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Final fusion layers
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(2*channels, 2*channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*channels, 2*channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*channels, channels, kernel_size=1)
        )
        
        # Skip connection
        self.skip_conv = nn.Conv2d(2*channels, channels, kernel_size=1)
        
    def forward(self, rgb_feat, chm_feat):
        """
        Args:
            rgb_feat: RGB features [N, C, H, W]
            chm_feat: CHM features [N, C, H, W]
        Returns:
            fused: Fused features [N, C, H, W]
        """
        # Cross-attention stage
        # Get Q,K,V for both modalities
        rgb_q = self.rgb_q(rgb_feat).flatten(2)  # [N, C/8, H*W]
        rgb_k = self.rgb_k(rgb_feat).flatten(2)  # [N, C/8, H*W]
        rgb_v = self.rgb_v(rgb_feat).flatten(2)  # [N, C, H*W]
        
        depth_q = self.depth_q(chm_feat).flatten(2)  # [N, C/8, H*W]
        depth_k = self.depth_k(chm_feat).flatten(2)  # [N, C/8, H*W]
        depth_v = self.depth_v(chm_feat).flatten(2)  # [N, C, H*W]
        
        # Compute attention maps
        # RGB enhanced by depth attention
        rgb_attn = torch.softmax(torch.bmm(depth_q.transpose(1,2), depth_k), dim=-1)  # [N, H*W, H*W]
        br = torch.bmm(rgb_v, rgb_attn)  # [N, C, H*W]
        br = br.view_as(rgb_feat) + rgb_feat  # [N, C, H, W]
        
        # Depth enhanced by RGB attention
        depth_attn = torch.softmax(torch.bmm(rgb_q.transpose(1,2), rgb_k), dim=-1)  # [N, H*W, H*W]
        bd = torch.bmm(depth_v, depth_attn)  # [N, C, H*W]
        bd = bd.view_as(chm_feat) + chm_feat  # [N, C, H, W]
        
        # Gated fusion stage
        # Concatenate features
        c = torch.cat([br, bd], dim=1)  # [N, 2C, H, W]
        
        # Compute gates
        g = self.gate_conv(c)  # [N, C, H, W]
        gr = g
        gd = 1 - g
        
        # Weighted sum
        c_fused = br * gr + bd * gd
        d_fused = br * bd
        
        # Final fusion
        f = torch.cat([c_fused, d_fused], dim=1)  # [N, 2C, H, W]
        fused = self.fusion_conv(f) + self.skip_conv(f)
        
        return fused