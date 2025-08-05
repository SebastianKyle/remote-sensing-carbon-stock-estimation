import torch
import torch.nn as nn
from torchvision import models
from torchvision.ops import misc
import sys
sys.path.append('../src')
from collections import OrderedDict
from models.attention.cross_attention import CrossAttention
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool

class CAResNet(nn.Module): 
    def __init__(self, pretrained=True, backbone_name='resnet50'):
        super(CAResNet, self).__init__()
        self.rgb_backbone = models.resnet.__dict__[backbone_name](
            pretrained=pretrained, norm_layer=misc.FrozenBatchNorm2d)
        self.chm_backbone = models.resnet.__dict__[backbone_name](
            pretrained=pretrained, norm_layer=misc.FrozenBatchNorm2d)

        self.rgb_layers = list(self.rgb_backbone.children())[:-2]
        self.chm_layers = list(self.chm_backbone.children())[:-2]
        
        self.cross_attention1 = CrossAttention(64)
        self.cross_attention2 = CrossAttention(256)
        self.cross_attention3 = CrossAttention(512)
        self.cross_attention4 = CrossAttention(1024)
        self.cross_attention5 = CrossAttention(2048)
        
        self.out_channels = 256
        self.inner_block_module = nn.Conv2d(2048, self.out_channels, 1)
        self.layer_block_module = nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1)

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
        fused_features["c2"] = self.cross_attention2(rgb_features, chm_features)
        
        # Stage 3
        rgb_features = self.rgb_layers[5](rgb_features)
        chm_features = self.chm_layers[5](chm_features)
        fused_features["c3"] = self.cross_attention3(rgb_features, chm_features)
        
        # Stage 4
        rgb_features = self.rgb_layers[6](rgb_features)
        chm_features = self.chm_layers[6](chm_features)
        fused_features["c4"] = self.cross_attention4(rgb_features, chm_features)

        # Stage 5
        rgb_features = self.rgb_layers[7](rgb_features)
        chm_features = self.chm_layers[7](chm_features)
        fused_features["c5"] = self.cross_attention5(rgb_features, chm_features)
         
        return self.fpn(fused_features)

# rgb_backbone = models.resnet.__dict__['resnet50'](pretrained=True, norm_layer=misc.FrozenBatchNorm2d)
# rgb_layers = list(rgb_backbone.children())[:-2]
# print(rgb_layers[7])