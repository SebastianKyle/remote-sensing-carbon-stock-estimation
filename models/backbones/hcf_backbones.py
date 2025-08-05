import torch
import torch.nn as nn
from torchvision import models
from torchvision.ops import misc
from collections import OrderedDict
from models.attention.cross_attention import CrossAttention
from models.attention.hcf import HCF, HCF_LKA, HCF_CA, HCF_RF
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool

class HCFResNet(nn.Module): 
    def __init__(self, pretrained=True, backbone_name='resnet50'):
        super(HCFResNet, self).__init__()
        self.rgb_backbone = models.resnet.__dict__[backbone_name](
            pretrained=pretrained, norm_layer=misc.FrozenBatchNorm2d)
        self.chm_backbone = models.resnet.__dict__[backbone_name](
            pretrained=False, norm_layer=misc.FrozenBatchNorm2d)

        for name, parameter in self.rgb_backbone.named_parameters(): 
            if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)

        for name, parameter in self.chm_backbone.named_parameters(): 
            if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)

        self.rgb_layers = list(self.rgb_backbone.children())[:-2]
        self.chm_layers = list(self.chm_backbone.children())[:-2]
        
        self.fusion2 = HCF(256)
        self.fusion3 = HCF(512)
        self.fusion4 = HCF(1024)
        self.fusion5 = HCF(2048)
        
        self.out_channels = 256

        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[256, 512, 1024, 2048],
            out_channels=self.out_channels,
            extra_blocks=LastLevelMaxPool()
        )

        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, rgb_image, chm_image):
        fused_features = {}

        # Stage 1
        rgb_features = self.rgb_layers[0](rgb_image)
        chm_features = self.chm_layers[0](chm_image)
        rgb_features = self.rgb_layers[1](rgb_features)
        chm_features = self.chm_layers[1](chm_features)
        rgb_features = self.rgb_layers[2](rgb_features)
        chm_features = self.chm_layers[2](chm_features)
        rgb_features = self.rgb_layers[3](rgb_features)
        chm_features = self.chm_layers[3](chm_features)
        
        # Stage 2
        rgb_features = self.rgb_layers[4](rgb_features)
        chm_features = self.chm_layers[4](chm_features)
        fused_features["c2"] = self.fusion2(rgb_features, chm_features)
        
        # Stage 3
        rgb_features = self.rgb_layers[5](rgb_features)
        chm_features = self.chm_layers[5](chm_features + fused_features["c2"])
        fused_features["c3"] = self.fusion3(rgb_features, chm_features)
        
        # Stage 4
        rgb_features = self.rgb_layers[6](rgb_features)
        chm_features = self.chm_layers[6](chm_features + fused_features["c3"])
        fused_features["c4"] = self.fusion4(rgb_features, chm_features)

        # Stage 5
        rgb_features = self.rgb_layers[7](rgb_features)
        chm_features = self.chm_layers[7](chm_features + fused_features["c4"])
        fused_features["c5"] = self.fusion5(rgb_features, chm_features)
         
        return self.fpn(fused_features)

class HCFResNetNoFPN(nn.Module): 
    def __init__(self, pretrained=True, backbone_name='resnet50'):
        super(HCFResNetNoFPN, self).__init__()
        self.rgb_backbone = models.resnet.__dict__[backbone_name](
            pretrained=pretrained, norm_layer=misc.FrozenBatchNorm2d)
        self.chm_backbone = models.resnet.__dict__[backbone_name](
            pretrained=False, norm_layer=misc.FrozenBatchNorm2d)

        for name, parameter in self.rgb_backbone.named_parameters(): 
            if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)

        for name, parameter in self.chm_backbone.named_parameters(): 
            if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)

        self.rgb_layers = list(self.rgb_backbone.children())[:-2]
        self.chm_layers = list(self.chm_backbone.children())[:-2]
        
        self.fusion2 = HCF(256)
        self.fusion3 = HCF(512)
        self.fusion4 = HCF(1024)
        self.fusion5 = HCF(2048)

        # Self-attention module to process final feature to have out channels = 256
        self.attn_fusion = nn.TransformerEncoderLayer(
            d_model=2048, 
            nhead=4, 
            batch_first=True
        )
        self.inner_block_module = nn.Conv2d(2048, 256, kernel_size=1)
        self.layer_block_module = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        
        self.out_channels = 256

        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, rgb_image, chm_image):
        # Stage 1
        rgb_features = self.rgb_layers[0](rgb_image)
        chm_features = self.chm_layers[0](chm_image)
        rgb_features = self.rgb_layers[1](rgb_features)
        chm_features = self.chm_layers[1](chm_features)
        rgb_features = self.rgb_layers[2](rgb_features)
        chm_features = self.chm_layers[2](chm_features)
        rgb_features = self.rgb_layers[3](rgb_features)
        chm_features = self.chm_layers[3](chm_features)
        
        # Stage 2
        rgb_features = self.rgb_layers[4](rgb_features)
        chm_features = self.chm_layers[4](chm_features)
        fused_2 = self.fusion2(rgb_features, chm_features)
        
        # Stage 3
        rgb_features = self.rgb_layers[5](rgb_features + fused_2)
        chm_features = self.chm_layers[5](chm_features + fused_2)
        fused_3 = self.fusion3(rgb_features, chm_features)
        
        # Stage 4
        rgb_features = self.rgb_layers[6](rgb_features + fused_3)
        chm_features = self.chm_layers[6](chm_features + fused_3)
        fused_4 = self.fusion4(rgb_features, chm_features)

        # Stage 5
        rgb_features = self.rgb_layers[7](rgb_features + fused_4)
        chm_features = self.chm_layers[7](chm_features + fused_4)
        fused_5 = self.fusion5(rgb_features, chm_features)

        # Apply self-attention to the final fused features
        B, C, H, W = fused_5.shape
        fused = fused_5.flatten(2).transpose(1, 2)  # [B, H*W, C]
        fused = self.attn_fusion(fused)  # Apply self-attention
        fused = fused.transpose(1, 2).view(B, C, H, W)  

        # Project to 256 channels
        fused = self.inner_block_module(fused)  
        fused = self.layer_block_module(fused)  

        fused_features = {}
        fused_features["c0"] = fused
         
        return fused_features

class HCFEfficientNet(nn.Module):
    def __init__(self, pretrained=True, backbone_name='efficientnet_v2_s'):
        super(HCFEfficientNet, self).__init__()

        self.rgb_backbone = models.efficientnet.__dict__[backbone_name](
            weights="DEFAULT" if pretrained else None,
        ).features

        self.chm_backbone = models.efficientnet.__dict__[backbone_name](
            weights="DEFAULT" if pretrained else None,
        ).features

        # EfficientNetV2-S stage-to-channel mapping:
        # features[2] → c2: 48
        # features[3] → c3: 64
        # features[4] → c4: 128
        # features[5] → c5: 160
        self.fusion2 = HCF(48)
        self.fusion3 = HCF(64)
        self.fusion4 = HCF(160)
        self.fusion5 = HCF(256)

        self.out_channels = 256
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[48, 64, 160, 256],
            out_channels=self.out_channels,
            extra_blocks=LastLevelMaxPool()
        )

        self.attn_fpn = SelfAttentionFPNWrapper(self.out_channels)

    def forward(self, rgb_image, chm_image):
        fused = {}

        # Stage 0, 1
        rgb = self.rgb_backbone[0](rgb_image)
        rgb = self.rgb_backbone[1](rgb)
        chm = self.chm_backbone[0](chm_image)
        chm = self.chm_backbone[1](chm)

        # Stage 2 → c2
        rgb = self.rgb_backbone[2](rgb)
        chm = self.chm_backbone[2](chm)
        fused["c2"] = self.fusion2(rgb, chm)

        # Stage 3 → c3
        rgb = self.rgb_backbone[3](rgb)
        chm = self.chm_backbone[3](chm + fused["c2"])
        fused["c3"] = self.fusion3(rgb, chm)

        # Stage 4 → c4
        rgb = self.rgb_backbone[5](self.rgb_backbone[4](rgb))
        chm = self.chm_backbone[5](self.chm_backbone[4](chm + fused["c3"]))
        fused["c4"] = self.fusion4(rgb, chm)

        # Stage 5 → c5
        rgb = self.rgb_backbone[6](rgb)
        chm = self.chm_backbone[6](chm + fused["c4"])
        fused["c5"] = self.fusion5(rgb, chm)

        fused = self.fpn(fused)
        fused = self.attn_fpn(fused)

        return fused

class HCFSwinEfficientNet(nn.Module):
    def __init__(self, pretrained=True):
        super(HCFSwinEfficientNet, self).__init__()

        # Swin RGB branch
        swin = models.swin_v2_s(weights="DEFAULT" if pretrained else None)
        self.rgb_stages = nn.ModuleList([
            swin.features[1],  # stage1 → c2
            nn.Sequential(swin.features[2], swin.features[3]),  # stage2 → c3
            nn.Sequential(swin.features[4], swin.features[5]),  # stage3 → c4
            nn.Sequential(swin.features[6], swin.features[7]),  # stage4 → c5
        ])
        self.rgb_patch_embed = swin.features[0]  # stem

        # EfficientNetV2 CHM branch
        eff = models.efficientnet_v2_s(weights="DEFAULT" if pretrained else None).features
        self.chm_layers = nn.ModuleList([
            eff[2],  # c3
            eff[3],  # c4
            nn.Sequential(eff[4], eff[5]),  # c5
            eff[6],  # c6
        ])
        self.chm_stem = nn.Sequential(eff[0], eff[1])  # pre-processing

        # Match Swin/EfficientNet dims
        self.chm_proj2 = nn.Conv2d(48, 96, kernel_size=1)
        self.chm_proj3 = nn.Conv2d(64, 192, kernel_size=1)
        self.chm_proj4 = nn.Conv2d(160, 384, kernel_size=1)
        self.chm_proj5 = nn.Conv2d(256, 768, kernel_size=1)

        self.chm_proj_back2 = nn.Conv2d(96, 48, kernel_size=1)
        self.chm_proj_back3 = nn.Conv2d(192, 64, kernel_size=1)
        self.chm_proj_back4 = nn.Conv2d(384, 160, kernel_size=1)

        # Fusion modules (all output 96–768 channels)
        self.fusion2 = HCF(96)
        self.fusion3 = HCF(192)
        self.fusion4 = HCF(384)
        self.fusion5 = HCF(768)

        self.out_channels = 256
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[96, 192, 384, 768],
            out_channels=self.out_channels,
            extra_blocks=LastLevelMaxPool()
        )

    def forward(self, rgb_image, chm_image):
        B = rgb_image.size(0)
        fused = {}

        # RGB forward through Swin stem
        rgb = self.rgb_patch_embed(rgb_image)  # [B, 96, H/4, W/4]

        # CHM stem
        chm = self.chm_stem(chm_image)

        # Stage 2
        rgb = self.rgb_stages[0](rgb)
        rgb_feat = rgb.permute(0, 3, 1, 2).contiguous()
        chm = self.chm_layers[0](chm)
        chm = self.chm_proj2(chm)
        fused["c2"] = self.fusion2(rgb_feat, chm)

        # Stage 3
        rgb = self.rgb_stages[1](rgb)
        rgb_feat = rgb.permute(0, 3, 1, 2).contiguous()
        chm = self.chm_layers[1](self.chm_proj_back2(chm + fused["c2"]))
        chm = self.chm_proj3(chm)
        fused["c3"] = self.fusion3(rgb_feat, chm)

        # Stage 4
        rgb = self.rgb_stages[2](rgb)
        rgb_feat = rgb.permute(0, 3, 1, 2).contiguous()
        chm = self.chm_layers[2](self.chm_proj_back3(chm + fused["c3"]))
        chm = self.chm_proj4(chm)
        fused["c4"] = self.fusion4(rgb_feat, chm)

        # Stage 5
        rgb = self.rgb_stages[3](rgb)
        rgb_feat = rgb.permute(0, 3, 1, 2).contiguous()
        chm = self.chm_layers[3](self.chm_proj_back4(chm + fused["c4"]))
        chm = self.chm_proj5(chm)
        fused["c5"] = self.fusion5(rgb_feat, chm)

        fused = self.fpn(fused)
        return fused

class HCFLKAResNet(nn.Module): 
    def __init__(self, pretrained=True, backbone_name='resnet50'):
        super(HCFLKAResNet, self).__init__()
        self.rgb_backbone = models.resnet.__dict__[backbone_name](
            pretrained=pretrained, norm_layer=misc.FrozenBatchNorm2d)
        self.chm_backbone = models.resnet.__dict__[backbone_name](
            pretrained=pretrained, norm_layer=misc.FrozenBatchNorm2d)

        for name, parameter in self.rgb_backbone.named_parameters(): 
            if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)

        for name, parameter in self.chm_backbone.named_parameters(): 
            if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)

        self.rgb_layers = list(self.rgb_backbone.children())[:-2]
        self.chm_layers = list(self.chm_backbone.children())[:-2]
        
        self.fusion2 = HCF_LKA(256)
        self.fusion3 = HCF_LKA(512)
        self.fusion4 = HCF_LKA(1024)
        self.fusion5 = HCF_LKA(2048)
        
        self.out_channels = 256

        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[256, 512, 1024, 2048],
            out_channels=self.out_channels,
            extra_blocks=LastLevelMaxPool()
        )

        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, rgb_image, chm_image):
        fused_features = {}

        # Stage 1
        rgb_features = self.rgb_layers[0](rgb_image)
        chm_features = self.chm_layers[0](chm_image)
        rgb_features = self.rgb_layers[1](rgb_features)
        chm_features = self.chm_layers[1](chm_features)
        rgb_features = self.rgb_layers[2](rgb_features)
        chm_features = self.chm_layers[2](chm_features)
        rgb_features = self.rgb_layers[3](rgb_features)
        chm_features = self.chm_layers[3](chm_features)
        
        # Stage 2
        rgb_features = self.rgb_layers[4](rgb_features)
        chm_features = self.chm_layers[4](chm_features)
        fused_features["c2"] = self.fusion2(rgb_features, chm_features)
        
        # Stage 3
        rgb_features = self.rgb_layers[5](rgb_features)
        chm_features = self.chm_layers[5](chm_features + fused_features["c2"])
        fused_features["c3"] = self.fusion3(rgb_features, chm_features)
        
        # Stage 4
        rgb_features = self.rgb_layers[6](rgb_features)
        chm_features = self.chm_layers[6](chm_features + fused_features["c3"])
        fused_features["c4"] = self.fusion4(rgb_features, chm_features)

        # Stage 5
        rgb_features = self.rgb_layers[7](rgb_features)
        chm_features = self.chm_layers[7](chm_features + fused_features["c4"])
        fused_features["c5"] = self.fusion5(rgb_features, chm_features)
         
        return self.fpn(fused_features)

class HCFCAResNet(nn.Module): 
    def __init__(self, pretrained=True, backbone_name='resnet50'):
        super(HCFCAResNet, self).__init__()
        self.rgb_backbone = models.resnet.__dict__[backbone_name](
            pretrained=pretrained, norm_layer=misc.FrozenBatchNorm2d)
        self.chm_backbone = models.resnet.__dict__[backbone_name](
            pretrained=pretrained, norm_layer=misc.FrozenBatchNorm2d)

        for name, parameter in self.rgb_backbone.named_parameters(): 
            if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)

        for name, parameter in self.chm_backbone.named_parameters(): 
            if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)

        self.rgb_layers = list(self.rgb_backbone.children())[:-2]
        self.chm_layers = list(self.chm_backbone.children())[:-2]
        
        self.fusion2 = HCF_CA(256)
        self.fusion3 = HCF_CA(512)
        self.fusion4 = HCF_CA(1024)
        self.fusion5 = HCF_CA(2048)
        
        self.out_channels = 256

        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[256, 512, 1024, 2048],
            out_channels=self.out_channels,
            extra_blocks=LastLevelMaxPool()
        )

        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, rgb_image, chm_image):
        fused_features = {}

        # Stage 1
        rgb_features = self.rgb_layers[0](rgb_image)
        chm_features = self.chm_layers[0](chm_image)
        rgb_features = self.rgb_layers[1](rgb_features)
        chm_features = self.chm_layers[1](chm_features)
        rgb_features = self.rgb_layers[2](rgb_features)
        chm_features = self.chm_layers[2](chm_features)
        rgb_features = self.rgb_layers[3](rgb_features)
        chm_features = self.chm_layers[3](chm_features)
        
        # Stage 2
        rgb_features = self.rgb_layers[4](rgb_features)
        chm_features = self.chm_layers[4](chm_features)
        fused_features["c2"] = self.fusion2(rgb_features, chm_features)
        
        # Stage 3
        rgb_features = self.rgb_layers[5](rgb_features)
        chm_features = self.chm_layers[5](chm_features + fused_features["c2"])
        fused_features["c3"] = self.fusion3(rgb_features, chm_features)
        
        # Stage 4
        rgb_features = self.rgb_layers[6](rgb_features)
        chm_features = self.chm_layers[6](chm_features + fused_features["c3"])
        fused_features["c4"] = self.fusion4(rgb_features, chm_features)

        # Stage 5
        rgb_features = self.rgb_layers[7](rgb_features)
        chm_features = self.chm_layers[7](chm_features + fused_features["c4"])
        fused_features["c5"] = self.fusion5(rgb_features, chm_features)
         
        return self.fpn(fused_features)

class HCFRFResNet(nn.Module): 
    def __init__(self, pretrained=True, backbone_name='resnet50'):
        super(HCFRFResNet, self).__init__()
        self.rgb_backbone = models.resnet.__dict__[backbone_name](
            pretrained=pretrained, norm_layer=misc.FrozenBatchNorm2d)
        self.chm_backbone = models.resnet.__dict__[backbone_name](
            pretrained=pretrained, norm_layer=misc.FrozenBatchNorm2d)

        for name, parameter in self.rgb_backbone.named_parameters(): 
            if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)

        for name, parameter in self.chm_backbone.named_parameters(): 
            if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)

        self.rgb_layers = list(self.rgb_backbone.children())[:-2]
        self.chm_layers = list(self.chm_backbone.children())[:-2]
        
        self.fusion2 = HCF_RF(256)
        self.fusion3 = HCF_RF(512)
        self.fusion4 = HCF_RF(1024)
        self.fusion5 = HCF_RF(2048)
        
        self.out_channels = 256

        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[256, 512, 1024, 2048],
            out_channels=self.out_channels,
            extra_blocks=LastLevelMaxPool()
        )

        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, rgb_image, chm_image):
        fused_features = {}

        # Stage 1
        rgb_features = self.rgb_layers[0](rgb_image)
        chm_features = self.chm_layers[0](chm_image)
        rgb_features = self.rgb_layers[1](rgb_features)
        chm_features = self.chm_layers[1](chm_features)
        rgb_features = self.rgb_layers[2](rgb_features)
        chm_features = self.chm_layers[2](chm_features)
        rgb_features = self.rgb_layers[3](rgb_features)
        chm_features = self.chm_layers[3](chm_features)
        
        # Stage 2
        rgb_features = self.rgb_layers[4](rgb_features)
        chm_features = self.chm_layers[4](chm_features)
        fused_features["c2"] = self.fusion2(rgb_features, chm_features)
        
        # Stage 3
        rgb_features = self.rgb_layers[5](rgb_features)
        chm_features = self.chm_layers[5](chm_features + fused_features["c2"])
        fused_features["c3"] = self.fusion3(rgb_features, chm_features)
        
        # Stage 4
        rgb_features = self.rgb_layers[6](rgb_features)
        chm_features = self.chm_layers[6](chm_features + fused_features["c3"])
        fused_features["c4"] = self.fusion4(rgb_features, chm_features)

        # Stage 5
        rgb_features = self.rgb_layers[7](rgb_features)
        chm_features = self.chm_layers[7](chm_features + fused_features["c4"])
        fused_features["c5"] = self.fusion5(rgb_features, chm_features)
         
        return self.fpn(fused_features)

class SelfAttentionFPNWrapper(nn.Module):
    def __init__(self, channels, heads=4):
        super().__init__()
        self.attn_blocks = nn.ModuleDict({
            name: nn.TransformerEncoderLayer(d_model=channels, nhead=heads, batch_first=True)
            for name in ['c2', 'c3', 'c4', 'c5', 'pool']
        })

    def forward(self, features):
        out = {}
        for k, feat in features.items():
            B, C, H, W = feat.shape
            x = feat.flatten(2).transpose(1, 2)  # [B, H*W, C]
            x = self.attn_blocks[k](x)
            x = x.transpose(1, 2).view(B, C, H, W)
            out[k] = x
        return out