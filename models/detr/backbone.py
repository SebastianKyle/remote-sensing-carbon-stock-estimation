import torch
import torch.nn as nn
import torchvision
from torchvision.models._utils import IntermediateLayerGetter
from utils.misc import NestedTensor, is_main_process
from typing import List
from .position_encoding import build_position_encoding
from models.backbones.cbam_resnet50 import CBAMResNet, CBAMResNetNoFPN
from collections import OrderedDict

class FrozenBatchNorm2d(nn.Module):
    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def forward(self, x):
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias

class BackboneBase(nn.Module):
    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = nn.functional.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out

class Backbone(BackboneBase):
    def __init__(self, name: str, train_backbone: bool, return_interm_layers: bool, dilation: bool):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)

class CustomBackboneBase(nn.Module): 
    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'c3' not in name and 'c4' not in name and 'c5' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"c2": "0", "c3": "1", "c4": "2", "c5": "3"}
        else:
            return_layers = {'c5': "0"}

        self.backbone = backbone
        self.return_layers = return_layers
        self.num_channels = num_channels

    def forward(self, rgb_tensor_list: NestedTensor, chm_tensor_list: NestedTensor):
        x = self.backbone(rgb_tensor_list.tensors, chm_tensor_list.tensors)
        xs = OrderedDict()
        for name, out_name in self.return_layers.items():
            xs[out_name] = x
        out = {}
        for name, x in xs.items():
            m = rgb_tensor_list.mask
            assert m is not None
            mask = nn.functional.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out

class CBAM_Backbone(CustomBackboneBase):
    def __init__(self, name: str, train_backbone: bool, return_interm_layers: bool, dilation: bool):
        backbone = CBAMResNetNoFPN(pretrained=is_main_process(), backbone_name=name) 
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers) 

class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos

class FusionJoiner(nn.Sequential): 
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, rgb_images: NestedTensor, chm_images: NestedTensor):
        xs = self[0](rgb_images, chm_images)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos

def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model