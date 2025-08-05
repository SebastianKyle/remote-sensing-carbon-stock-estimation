from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.model_zoo import load_url
from torchvision import models
from torchvision.ops import misc
from torchvision.ops import MultiScaleRoIAlign

import sys

import models.backbones
sys.path.append('../src')

import models
from models.rcnn.utils import AnchorGenerator, MultiScaleAnchorGenerator
from models.rcnn.rpn import RPNHead, RegionProposalNetwork, RegionProposalNetworkSingleFeature
from models.rcnn.pooler import RoIAlign
from models.rcnn.roi_heads import RoIHeads
from models.rcnn.transform import Transformer
from models.rcnn.faster_rcnn import FastRCNNPredictor

from models.backbones.cbam_resnet50 import CBAMResNet

class FusedRCNN(nn.Module):
    
    def __init__(self, backbone, num_classes,
                 # RPN parameters
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_num_samples=256, rpn_positive_fraction=0.5,
                 rpn_reg_weights=(1., 1., 1., 1.),
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                 rpn_nms_thresh=0.7,
                 # RoIHeads parameters
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                 box_num_samples=512, box_positive_fraction=0.25,
                 box_reg_weights=(10., 10., 5., 5.),
                 box_score_thresh=0.1, box_nms_thresh=0.2, box_num_detections=300,
                 anchor_sizes=((32,), (64,), (128,), (256,), (512,)),
                 loss_cfg=None,
                 cfg=None
                 ):
        super().__init__()
        self.backbone = backbone
        out_channels = backbone.out_channels
        
        #------------- RPN --------------------------
        # anchor_sizes = ((16,), (32,), (64,), (128,), (256,))
        # anchor_sizes = ((16,), (32,), (64,), (128,), (256,), (512,))
        # anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
        anchor_ratios = (0.5, 1, 2)
        num_anchors = len(anchor_ratios)
        rpn_anchor_generator = MultiScaleAnchorGenerator(anchor_sizes, anchor_ratios)
        rpn_head = RPNHead(out_channels, num_anchors)
        
        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)
        self.rpn = RegionProposalNetwork(
             rpn_anchor_generator, rpn_head, 
             rpn_fg_iou_thresh, rpn_bg_iou_thresh,
             rpn_num_samples, rpn_positive_fraction,
             rpn_reg_weights,
             rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh)
        
        #------------ RoIHeads --------------------------
        # box_roi_pool = MultiScaleRoIAlign(featmap_names=["c2", "c3", "c4", "c5", "pool"], output_size=(7, 7), sampling_ratio=2)
        box_roi_pool = MultiScaleRoIAlign(featmap_names=["c2", "c3", "c4", "c5"], output_size=(7, 7), sampling_ratio=2)
        
        resolution = box_roi_pool.output_size[0]
        in_channels = out_channels * resolution ** 2
        mid_channels = 1024
        box_predictor = FastRCNNPredictor(in_channels, mid_channels, num_classes)
        
        self.head = RoIHeads(
             box_roi_pool, box_predictor,
             box_fg_iou_thresh, box_bg_iou_thresh,
             box_num_samples, box_positive_fraction,
             box_reg_weights,
             box_score_thresh, box_nms_thresh, box_num_detections)
        if loss_cfg is not None:
            self.head.loss_cfg = loss_cfg

        #------------ Transformer --------------------------
        self.transformer = Transformer(
            min_size=400, max_size=400, 
            image_mean=[0.485, 0.456, 0.406], 
            image_std=[0.229, 0.224, 0.225],
            resize_to=(400, 400))

        self.chm_transformer = Transformer(
            min_size=400, max_size=400,
            image_mean=[0.0, 0.0, 0.0],
            image_std=[1.0, 1.0, 1.0],
            resize_to=(400, 400) 
        )

    def forward(self, rgb_images, chm_images, targets=None):

        if not isinstance(rgb_images, (list, tuple)):
            rgb_images = [rgb_images]
        if not isinstance(chm_images, (list, tuple)):
            chm_images = [chm_images]
        
        original_image_sizes = []
        for img in rgb_images:
            val = img.shape[-2:]
            original_image_sizes.append((val[0], val[1]))

        rgb_images, targets = self.transformer(rgb_images, targets)
        # chm_images, _ = self.chm_transformer(chm_images, None)
        chm_images, _ = self.transformer(chm_images, None)
        
        if isinstance(rgb_images, (list, tuple)):
            rgb_images = torch.stack(rgb_images)
        if isinstance(chm_images, (list, tuple)):
            chm_images = torch.stack(chm_images)

        features = self.backbone(rgb_images, chm_images)
        
        if isinstance(features, torch.Tensor):
            features = {'0': features}
        
        # Get image shapes for each image in the batch
        image_shapes = [(img.shape[-2], img.shape[-1]) for img in rgb_images]
        
        proposals, rpn_losses = self.rpn(features, image_shapes, targets)
        detections, detector_losses = self.head(features, proposals, image_shapes, targets)
        
        detections = self.transformer.postprocess(detections, image_shapes, original_image_sizes)
        
        losses = {}
        if self.training:
            losses.update(rpn_losses)
            losses.update(detector_losses)
            return losses
            
        return detections

class FusedRCNNNoFPN(nn.Module):
    
    def __init__(self, backbone, num_classes,
                 # RPN parameters
                 rpn_fg_iou_thresh=0.5, rpn_bg_iou_thresh=0.3,
                 rpn_num_samples=256, rpn_positive_fraction=0.6,
                 rpn_reg_weights=(1., 1., 1., 1.),
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                 rpn_nms_thresh=0.7,
                 # RoIHeads parameters
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                 box_num_samples=512, box_positive_fraction=0.4,
                 box_reg_weights=(10., 10., 5., 5.),
                 box_score_thresh=0.1, box_nms_thresh=0.2, box_num_detections=300):
        super().__init__()
        self.backbone = backbone
        out_channels = backbone.out_channels
        
        #------------- RPN --------------------------
        anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
        anchor_ratios = (0.5, 1, 2)
        num_anchors = len(anchor_ratios)
        rpn_anchor_generator = MultiScaleAnchorGenerator(anchor_sizes, anchor_ratios)
        rpn_head = RPNHead(out_channels, num_anchors)
        
        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)
        self.rpn = RegionProposalNetwork(
             rpn_anchor_generator, rpn_head, 
             rpn_fg_iou_thresh, rpn_bg_iou_thresh,
             rpn_num_samples, rpn_positive_fraction,
             rpn_reg_weights,
             rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh)
        
        #------------ RoIHeads --------------------------
        box_roi_pool = MultiScaleRoIAlign(featmap_names=["c0"], output_size=(7, 7), sampling_ratio=2)
        
        resolution = box_roi_pool.output_size[0]
        in_channels = out_channels * resolution ** 2
        mid_channels = 1024
        box_predictor = FastRCNNPredictor(in_channels, mid_channels, num_classes)
        
        self.head = RoIHeads(
             box_roi_pool, box_predictor,
             box_fg_iou_thresh, box_bg_iou_thresh,
             box_num_samples, box_positive_fraction,
             box_reg_weights,
             box_score_thresh, box_nms_thresh, box_num_detections)

        #------------ Transformer --------------------------
        self.transformer = Transformer(
            min_size=400, max_size=400, 
            image_mean=[0.485, 0.456, 0.406], 
            image_std=[0.229, 0.224, 0.225],
            resize_to=(400, 400))

        self.chm_transformer = Transformer(
            min_size=400, max_size=400,
            image_mean=[0.0, 0.0, 0.0],
            image_std=[1.0, 1.0, 1.0],
            resize_to=(400, 400) 
        )

    def forward(self, rgb_images, chm_images, targets=None):

        if not isinstance(rgb_images, (list, tuple)):
            rgb_images = [rgb_images]
        if not isinstance(chm_images, (list, tuple)):
            chm_images = [chm_images]
        
        original_image_sizes = []
        for img in rgb_images:
            val = img.shape[-2:]
            original_image_sizes.append((val[0], val[1]))

        rgb_images, targets = self.transformer(rgb_images, targets)
        chm_images, _ = self.chm_transformer(chm_images, None)
        
        if isinstance(rgb_images, (list, tuple)):
            rgb_images = torch.stack(rgb_images)
        if isinstance(chm_images, (list, tuple)):
            chm_images = torch.stack(chm_images)

        features = self.backbone(rgb_images, chm_images)
        
        if isinstance(features, torch.Tensor):
            features = {'0': features}
        
        # Get image shapes for each image in the batch
        image_shapes = [(img.shape[-2], img.shape[-1]) for img in rgb_images]
        
        proposals, rpn_losses = self.rpn(features, image_shapes, targets)
        detections, detector_losses = self.head(features, proposals, image_shapes, targets)
        
        detections = self.transformer.postprocess(detections, image_shapes, original_image_sizes)
        
        losses = {}
        if self.training:
            losses.update(rpn_losses)
            losses.update(detector_losses)
            return losses
            
        return detections

def fused_rcnn_resnet50(pretrained, num_classes, backbone_name='cbam_resnet50', pretrained_chm_backbone=True, anchor_sizes=((32,), (64,), (128,), (256,), (512,)), loss_cfg=None):
    # Get the backbone class from the backbones dictionary
    backbone = models.backbones.backbones[backbone_name](pretrained=pretrained)
    model = FusedRCNN(backbone, num_classes=num_classes, anchor_sizes=anchor_sizes, loss_cfg=loss_cfg)

    if pretrained:
        model_urls = {
            'fasterrcnn_resnet50_fpn_coco':
                'https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth'
        }
        model_state_dict = load_url(model_urls['fasterrcnn_resnet50_fpn_coco'])
        
        pretrained_msd = list(model_state_dict.values())
        del_list = [i for i in range(265, 271)] + [i for i in range(273, 279)]
        for i, del_idx in enumerate(del_list):
            pretrained_msd.pop(del_idx - i)

        msd = model.state_dict()
        skip_list = ['backbone.rgb_backbone.fc.weight', 'backbone.rgb_backbone.fc.bias', 'backbone.inner_block_module.weight', 'backbone.inner_block_module.bias', 'backbone.layer_block_module.weight', 'backbone.layer_block_module.bias']
        if num_classes == 91:
            skip_list = [271, 272, 273, 274]

        for i, name in enumerate(msd):
            if name.startswith('backbone.rgb_backbone') and name not in skip_list and not name.startswith('backbone.fusion'):
                pretrained_name = name.replace('backbone.rgb_backbone', 'backbone.body')
                if pretrained_name in model_state_dict:
                    msd[name].copy_(model_state_dict[pretrained_name])

        if pretrained_chm_backbone:
            for i, name in enumerate(msd):
                if name.startswith('backbone.chm_backbone') and name not in skip_list and not name.startswith('backbone.fusion'):
                    pretrained_name = name.replace('backbone.chm_backbone', 'backbone.body')
                    if pretrained_name in model_state_dict:
                        msd[name].copy_(model_state_dict[pretrained_name])

        msd['backbone.fpn.inner_blocks.0.0.weight'].copy_(model_state_dict['backbone.fpn.inner_blocks.0.weight'])
        msd['backbone.fpn.inner_blocks.0.0.bias'].copy_(model_state_dict['backbone.fpn.inner_blocks.0.bias'])
        msd['backbone.fpn.inner_blocks.1.0.weight'].copy_(model_state_dict['backbone.fpn.inner_blocks.1.weight'])
        msd['backbone.fpn.inner_blocks.1.0.bias'].copy_(model_state_dict['backbone.fpn.inner_blocks.1.bias'])
        msd['backbone.fpn.inner_blocks.2.0.weight'].copy_(model_state_dict['backbone.fpn.inner_blocks.2.weight'])
        msd['backbone.fpn.inner_blocks.2.0.bias'].copy_(model_state_dict['backbone.fpn.inner_blocks.2.bias'])
        msd['backbone.fpn.inner_blocks.3.0.weight'].copy_(model_state_dict['backbone.fpn.inner_blocks.3.weight'])
        msd['backbone.fpn.inner_blocks.3.0.bias'].copy_(model_state_dict['backbone.fpn.inner_blocks.3.bias'])

        msd['backbone.fpn.layer_blocks.0.0.weight'].copy_(model_state_dict['backbone.fpn.layer_blocks.0.weight'])
        msd['backbone.fpn.layer_blocks.0.0.bias'].copy_(model_state_dict['backbone.fpn.layer_blocks.0.bias'])
        msd['backbone.fpn.layer_blocks.1.0.weight'].copy_(model_state_dict['backbone.fpn.layer_blocks.1.weight'])
        msd['backbone.fpn.layer_blocks.1.0.bias'].copy_(model_state_dict['backbone.fpn.layer_blocks.1.bias'])
        msd['backbone.fpn.layer_blocks.2.0.weight'].copy_(model_state_dict['backbone.fpn.layer_blocks.2.weight'])
        msd['backbone.fpn.layer_blocks.2.0.bias'].copy_(model_state_dict['backbone.fpn.layer_blocks.2.bias'])
        msd['backbone.fpn.layer_blocks.3.0.weight'].copy_(model_state_dict['backbone.fpn.layer_blocks.3.weight'])
        msd['backbone.fpn.layer_blocks.3.0.bias'].copy_(model_state_dict['backbone.fpn.layer_blocks.3.bias'])

        for i, name in enumerate(msd):
            if not name.startswith('backbone.rgb_backbone') and not name.startswith('backbone.chm_backbone') and name not in skip_list and not name.startswith('backbone.fusion'):
                if name in model_state_dict:
                    msd[name].copy_(model_state_dict[name])

        msd['head.box_predictor.fc1.weight'].copy_(model_state_dict['roi_heads.box_head.fc6.weight'])
        msd['head.box_predictor.fc1.bias'].copy_(model_state_dict['roi_heads.box_head.fc6.bias'])
        msd['head.box_predictor.fc2.weight'].copy_(model_state_dict['roi_heads.box_head.fc7.weight'])
        msd['head.box_predictor.fc2.bias'].copy_(model_state_dict['roi_heads.box_head.fc7.bias'])
            
        model.load_state_dict(msd)
    
    return model

def fused_rcnn_resnet50_no_fpn(pretrained, num_classes, backbone_name='hcf_resnet50_nofpn', pretrained_chm_backbone=True):
    # Get the backbone class from the backbones dictionary
    backbone = models.backbones.backbones[backbone_name](pretrained=pretrained)
    model = FusedRCNNNoFPN(backbone, num_classes=num_classes)

    if pretrained:
        model_urls = {
            'fasterrcnn_resnet50_fpn_coco':
                'https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth'
        }
        model_state_dict = load_url(model_urls['fasterrcnn_resnet50_fpn_coco'])
        
        pretrained_msd = list(model_state_dict.values())
        del_list = [i for i in range(265, 271)] + [i for i in range(273, 279)]
        for i, del_idx in enumerate(del_list):
            pretrained_msd.pop(del_idx - i)

        msd = model.state_dict()
        skip_list = ['backbone.rgb_backbone.fc.weight', 'backbone.rgb_backbone.fc.bias', 'backbone.inner_block_module.weight', 'backbone.inner_block_module.bias', 'backbone.layer_block_module.weight', 'backbone.layer_block_module.bias']
        if num_classes == 91:
            skip_list = [271, 272, 273, 274]

        for i, name in enumerate(msd):
            if name.startswith('backbone.rgb_backbone') and name not in skip_list and not name.startswith('backbone.fusion'):
                pretrained_name = name.replace('backbone.rgb_backbone', 'backbone.body')
                if pretrained_name in model_state_dict:
                    msd[name].copy_(model_state_dict[pretrained_name])

        if pretrained_chm_backbone:
            for i, name in enumerate(msd):
                if name.startswith('backbone.chm_backbone') and name not in skip_list and not name.startswith('backbone.fusion'):
                    pretrained_name = name.replace('backbone.chm_backbone', 'backbone.body')
                    if pretrained_name in model_state_dict:
                        msd[name].copy_(model_state_dict[pretrained_name])

        msd['head.box_predictor.fc1.weight'].copy_(model_state_dict['roi_heads.box_head.fc6.weight'])
        msd['head.box_predictor.fc1.bias'].copy_(model_state_dict['roi_heads.box_head.fc6.bias'])
        msd['head.box_predictor.fc2.weight'].copy_(model_state_dict['roi_heads.box_head.fc7.weight'])
        msd['head.box_predictor.fc2.bias'].copy_(model_state_dict['roi_heads.box_head.fc7.bias'])
            
        model.load_state_dict(msd)
    
    return model


def fused_rcnn_efficientnet_v2_s(pretrained, num_classes, backbone_name='hcf_efficientnet'):
    # Instantiate backbone
    backbone = models.backbones.backbones[backbone_name](pretrained=pretrained)
    model = FusedRCNN(backbone, num_classes=num_classes)

    if pretrained:
        model_urls = {
            'fasterrcnn_resnet50_fpn_coco': 
                'https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth'
        }

        state_dict = load_url(model_urls['fasterrcnn_resnet50_fpn_coco'])

        msd = model.state_dict()

        # Load matching FPN weights (inner_blocks, layer_blocks)
        # fpn_keys = [
        #     'backbone.fpn.inner_blocks.0.0.weight',
        #     'backbone.fpn.inner_blocks.0.0.bias',
        #     'backbone.fpn.inner_blocks.1.0.weight',
        #     'backbone.fpn.inner_blocks.1.0.bias',
        #     'backbone.fpn.inner_blocks.2.0.weight',
        #     'backbone.fpn.inner_blocks.2.0.bias',
        #     'backbone.fpn.inner_blocks.3.0.weight',
        #     'backbone.fpn.inner_blocks.3.0.bias',
        #     'backbone.fpn.layer_blocks.0.0.weight',
        #     'backbone.fpn.layer_blocks.0.0.bias',
        #     'backbone.fpn.layer_blocks.1.0.weight',
        #     'backbone.fpn.layer_blocks.1.0.bias',
        #     'backbone.fpn.layer_blocks.2.0.weight',
        #     'backbone.fpn.layer_blocks.2.0.bias',
        #     'backbone.fpn.layer_blocks.3.0.weight',
        #     'backbone.fpn.layer_blocks.3.0.bias',
        # ]
        # for k in fpn_keys:
        #     if k in msd and k.replace('.0.', '.') in state_dict:
        #         msd[k].copy_(state_dict[k.replace('.0.', '.')])

        # Load ROI head weights
        msd['head.box_predictor.fc1.weight'].copy_(state_dict['roi_heads.box_head.fc6.weight'])
        msd['head.box_predictor.fc1.bias'].copy_(state_dict['roi_heads.box_head.fc6.bias'])
        msd['head.box_predictor.fc2.weight'].copy_(state_dict['roi_heads.box_head.fc7.weight'])
        msd['head.box_predictor.fc2.bias'].copy_(state_dict['roi_heads.box_head.fc7.bias'])

        model.load_state_dict(msd)

    return model

def fused_rcnn_convnext(pretrained, num_classes, backbone_name='cbam_convnext'):
    # Get the backbone class from the backbones dictionary
    backbone = models.backbones.backbones[backbone_name](pretrained=pretrained)
    model = FusedRCNN(backbone, num_classes=num_classes)

    if pretrained:
        model_urls = {
            'fasterrcnn_resnet50_fpn_coco':
                'https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth'
        }
        model_state_dict = load_url(model_urls['fasterrcnn_resnet50_fpn_coco'])
        
        pretrained_msd = list(model_state_dict.values())
        del_list = [i for i in range(265, 271)] + [i for i in range(273, 279)]
        for i, del_idx in enumerate(del_list):
            pretrained_msd.pop(del_idx - i)

        msd = model.state_dict()
        skip_list = ['backbone.rgb_backbone.fc.weight', 'backbone.rgb_backbone.fc.bias', 'backbone.inner_block_module.weight', 'backbone.inner_block_module.bias', 'backbone.layer_block_module.weight', 'backbone.layer_block_module.bias']
        if num_classes == 91:
            skip_list = [271, 272, 273, 274]

        for i, name in enumerate(msd):
            if not name.startswith('backbone.rgb_backbone') and not name.startswith('backbone.chm_backbone') and name not in skip_list and not name.startswith('backbone.fusion'):
                if name in model_state_dict:
                    msd[name].copy_(model_state_dict[name])

        msd['head.box_predictor.fc1.weight'].copy_(model_state_dict['roi_heads.box_head.fc6.weight'])
        msd['head.box_predictor.fc1.bias'].copy_(model_state_dict['roi_heads.box_head.fc6.bias'])
        msd['head.box_predictor.fc2.weight'].copy_(model_state_dict['roi_heads.box_head.fc7.weight'])
        msd['head.box_predictor.fc2.bias'].copy_(model_state_dict['roi_heads.box_head.fc7.bias'])
            
        model.load_state_dict(msd)
    
    return model