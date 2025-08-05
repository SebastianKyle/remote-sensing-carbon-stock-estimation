import torch
import torch.nn as nn
import timm
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool
from models.attention.cbam import CBAMFusion1

class CBAMConvNeXt(nn.Module):
    def __init__(self, pretrained=True, backbone_name='convnext_base'):  
        super(CBAMConvNeXt, self).__init__()
        self.rgb_backbone = timm.create_model(backbone_name, pretrained=pretrained, features_only=True, out_indices=(0, 1, 2, 3))
        self.chm_backbone = timm.create_model(backbone_name, pretrained=pretrained, features_only=True, out_indices=(0, 1, 2, 3))
        
        self.fusion2 = CBAMFusion1(128)
        self.fusion3 = CBAMFusion1(256)
        self.fusion4 = CBAMFusion1(512)
        self.fusion5 = CBAMFusion1(1024)
        
        self.out_channels = 256
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[128, 256, 512, 1024],
            out_channels=self.out_channels,
            extra_blocks=LastLevelMaxPool()
        )

    def forward(self, rgb_image, chm_image):
        rgb_features = self.rgb_backbone(rgb_image)
        chm_features = self.chm_backbone(chm_image)
        
        fused_features = {
            "c2": self.fusion2(rgb_features[0], chm_features[0]),
            "c3": self.fusion3(rgb_features[1], chm_features[1]),
            "c4": self.fusion4(rgb_features[2], chm_features[2]),
            "c5": self.fusion5(rgb_features[3], chm_features[3])
        }
        
        return self.fpn(fused_features)