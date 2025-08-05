import torch
import torch.nn as nn
from .conv import Conv
from .block import C3k2, A2C2f
from .modules import Concat
from models.attention.cbam import CBAMFusion1
from models.attention.hcf import HCF

class Backbone(nn.Module):
    """YOLOv12 Backbone (R-ELAN)"""
    def __init__(self, in_channels=3, width_multiple=0.25, depth_multiple=0.50, max_channels=1024):
        super().__init__()
        ch = [64, 128, 256, 512, 1024]
        ch = [min(int(c * width_multiple), max_channels) for c in ch]  # Ensure max_channels is 256
        n = [max(round(d * depth_multiple), 1) for d in [2, 2, 4, 4, 2]]  # Depth scaling

        self.stem = Conv(in_channels, ch[0], 3, 2, 1)  # Initial convolution
        self.stage1 = nn.Sequential(
            Conv(ch[0], ch[1], 3, 2, 1),
            C3k2(ch[1], ch[2], n[0], e=0.25, c3k=True)
        )
        self.stage2 = nn.Sequential(
            Conv(ch[2], ch[2], 3, 2, 1),
            C3k2(ch[2], ch[3], n[1], e=0.25, c3k=True)
        )
        self.stage3 = nn.Sequential(
            Conv(ch[3], ch[3], 3, 2, 1),
            A2C2f(ch[3], ch[3], n[2], a2=True, area=4)
        )
        self.stage4 = nn.Sequential(
            Conv(ch[3], ch[4], 3, 2, 1),
            A2C2f(ch[4], ch[4], n[3], a2=True, area=1)
        )

    def forward(self, x):
        p1 = self.stem(x)
        p2 = self.stage1(p1) # 32
        p3 = self.stage2(p2) # 64
        p4 = self.stage3(p3) # 128
        p5 = self.stage4(p4) # 256

        return p3, p4, p5  # Feature maps for P3, P4, P5

class FusionBackbone(nn.Module): 
    """YOLOv12 Fusion Backbone (R-ELAN)"""
    def __init__(self, in_channels=3, width_multiple=0.25, depth_multiple=0.50, max_channels=1024):
        super().__init__()
        ch = [64, 128, 256, 512, 1024]
        ch = [min(int(c * width_multiple), max_channels) for c in ch]  # Ensure max_channels is 256
        n = [max(round(d * depth_multiple), 1) for d in [2, 2, 4, 4, 2]]  # Depth scaling

        self.rgb_stem = Conv(in_channels, ch[0], 3, 2, 1)
        self.rgb_stage1 = nn.Sequential(
            Conv(ch[0], ch[1], 3, 2, 1),
            C3k2(ch[1], ch[2], n[0], e=0.25, c3k=True)
        )
        self.rgb_stage2 = nn.Sequential(
            Conv(ch[2], ch[2], 3, 2, 1),
            C3k2(ch[2], ch[3], n[1], e=0.25, c3k=True)
        )
        self.rgb_stage3 = nn.Sequential(
            Conv(ch[3], ch[3], 3, 2, 1),
            A2C2f(ch[3], ch[3], n[2], a2=True, area=4)
        )
        self.rgb_stage4 = nn.Sequential(
            Conv(ch[3], ch[4], 3, 2, 1),
            A2C2f(ch[4], ch[4], n[3], a2=True, area=1)
        )

        self.chm_stem = Conv(in_channels, ch[0], 3, 2, 1)
        self.chm_stage1 = nn.Sequential(
            Conv(ch[0], ch[1], 3, 2, 1),
            C3k2(ch[1], ch[2], n[0], e=0.25, c3k=True)
        )
        self.chm_stage2 = nn.Sequential(
            Conv(ch[2], ch[2], 3, 2, 1),
            C3k2(ch[2], ch[3], n[1], e=0.25, c3k=True)
        )
        self.chm_stage3 = nn.Sequential(
            Conv(ch[3], ch[3], 3, 2, 1),
            A2C2f(ch[3], ch[3], n[2], a2=True, area=4)
        )
        self.chm_stage4 = nn.Sequential(
            Conv(ch[3], ch[4], 3, 2, 1),
            A2C2f(ch[4], ch[4], n[3], a2=True, area=1)
        )

        self.fusion3 = HCF(ch[3])
        self.fusion4 = HCF(ch[3])
        self.fusion5 = HCF(ch[4])

    def forward(self, rgb, chm):
        rgb_p1 = self.rgb_stem(rgb)
        rgb_p2 = self.rgb_stage1(rgb_p1)
        rgb_p3 = self.rgb_stage2(rgb_p2)
        rgb_p4 = self.rgb_stage3(rgb_p3)
        rgb_p5 = self.rgb_stage4(rgb_p4)

        chm_p1 = self.chm_stem(chm)
        chm_p2 = self.chm_stage1(chm_p1)
        chm_p3 = self.chm_stage2(chm_p2)
        chm_p4 = self.chm_stage3(chm_p3)
        chm_p5 = self.chm_stage4(chm_p4)
        
        f3 = self.fusion3(rgb_p3, chm_p3)
        f4 = self.fusion4(rgb_p4, chm_p4)
        f5 = self.fusion5(rgb_p5, chm_p5)
        
        return f3, f4, f5  # Feature maps for P3, P4, P5 

class Neck(nn.Module):
    """YOLOv12 Neck with Feature Fusion"""
    def __init__(self, width_multiple=0.25, depth_multiple=0.50, max_channels=1024):
        super().__init__()
        ch = [64, 128, 256, 512, 1024]
        ch = [min(int(c * width_multiple), max_channels) for c in ch]  # Ensure max_channels is 256
        n = [max(round(d * depth_multiple), 1) for d in [2, 2, 2, 2]]  # Depth scaling

        self.upsample1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.concat1 = Concat()
        self.merge1 = A2C2f(ch[4] + ch[3], ch[3], n[0], a2=False)  # Merge P5 with P4

        self.upsample2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.concat2 = Concat()
        self.merge2 = A2C2f(ch[3] + ch[3], ch[2], n[1], a2=False)  # Merge P4 with P3

        self.downsample1 = Conv(ch[2], ch[2], 3, 2, 1)
        self.concat3 = Concat()
        self.merge3 = A2C2f(ch[3] + ch[2], ch[3], n[2], a2=False)  # Merge P4 with P3

        self.downsample2 = Conv(ch[3], ch[3], 3, 2, 1)
        self.concat4 = Concat()
        self.merge4 = C3k2(ch[4] + ch[3], ch[4], n[3], c3k=True)  # Merge P5 with P4

    def forward(self, p3, p4, p5):
        p5_up = self.upsample1(p5)
        p4_merge = self.merge1(self.concat1([p5_up, p4]))

        p4_up = self.upsample2(p4_merge)
        p3_merge = self.merge2(self.concat2([p4_up, p3]))

        p3_down = self.downsample1(p3_merge)
        p4_merge = self.merge3(self.concat3([p3_down, p4_merge]))

        p4_down = self.downsample2(p4_merge)
        p5_merge = self.merge4(self.concat4([p4_down, p5]))

        return p3_merge, p4_merge, p5_merge  # Output features for detection