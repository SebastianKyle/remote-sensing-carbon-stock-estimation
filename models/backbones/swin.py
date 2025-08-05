import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool
from collections import OrderedDict
from timm.models.swin_transformer import SwinTransformer

class SwinTransformerBackbone(nn.Module):
    def __init__(self, pretrained=True, backbone_name='swin_base_patch4_window12_384'):
        super(SwinTransformerBackbone, self).__init__()
        # Initialize Swin Transformer with base configuration
        self.swin = SwinTransformer(
            img_size=384,  # This will be adjusted dynamically
            patch_size=4,
            in_chans=3,
            num_classes=1000,
            embed_dim=128,
            depths=(2, 2, 18, 2),
            num_heads=(4, 8, 16, 32),
            window_size=12,
            mlp_ratio=4.,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            norm_layer=nn.LayerNorm,
            ape=False,
            patch_norm=True,
            pretrained=pretrained,
            pretrained_cfg=backbone_name,
            pretrained_cfg_overlay={
                'url': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth',
                'input_size': (3, 384, 384),
                'pool_size': None,
                'crop_pct': 1.0,
                'interpolation': 'bicubic',
                'mean': (0.485, 0.456, 0.406),
                'std': (0.229, 0.224, 0.225),
                'classifier': 'head'
            }
        )
        
        # Remove the classification head
        self.swin.head = nn.Identity()
        
        # Define output channels for FPN
        self.out_channels = 256
        
        # Add 1x1 convolutions to adjust channel dimensions
        self.channel_adjust = nn.ModuleDict({
            'c2': nn.Conv2d(128, 256, 1),
            'c3': nn.Conv2d(256, 512, 1),
            'c4': nn.Conv2d(512, 1024, 1),
            'c5': nn.Conv2d(1024, 2048, 1)
        })
        
        # Feature Pyramid Network
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[256, 512, 1024, 2048],  # Match ResNet output channels
            out_channels=self.out_channels,
            extra_blocks=LastLevelMaxPool()
        )
        
        # Initialize weights
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Get intermediate features from Swin Transformer
        features = {}
        
        # Get input dimensions
        # B, C, H, W = x.shape
        
        # Adjust patch embedding for the input size
        # self.swin.patch_embed.img_size = (H, W)
        # self.swin.patch_embed.grid_size = (H // 4, W // 4)  # patch_size=4
        # self.swin.patch_embed.num_patches = (H // 4) * (W // 4)
        
        # Patch embedding and positional embedding
        x = self.swin.patch_embed(x)  # B, L, C
        
        # Process through Swin Transformer stages
        for i, layer in enumerate(self.swin.layers):
            x = layer(x)
            # Handle 4D tensor [B, H, W, C]
            if isinstance(x, tuple):  # Some layers return tuple of (output, aux_output)
                x = x[0]
            # Transpose to [B, C, H, W]
            x_2d = x.permute(0, 3, 1, 2)
            # Adjust channels to match ResNet dimensions
            x_2d = self.channel_adjust[f'c{i+2}'](x_2d)
            features[f"c{i+2}"] = x_2d  # c2, c3, c4, c5
        
        return self.fpn(features)  # Returns P2, P3, P4, P5, P6