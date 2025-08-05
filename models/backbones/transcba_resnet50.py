import torch
import torch.nn as nn
from torchvision import models
from torchvision.ops import misc
from collections import OrderedDict
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool
from models.attention.trans_cbam import TransCBA

class TransCBAResNet(nn.Module): 
    def __init__(self, pretrained=True, backbone_name='resnet50'):
        super(TransCBAResNet, self).__init__()
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
        
        self.fusion2 = TransCBA(256, num_heads=8)
        self.fusion3 = TransCBA(512, num_heads=4)
        self.fusion4 = TransCBA(1024, num_heads=2)
        self.fusion5 = TransCBA(2048, num_heads=1)
        
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