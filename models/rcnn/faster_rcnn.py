from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.model_zoo import load_url
from torchvision import models
from torchvision.ops import misc
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool
from torchvision.ops import MultiScaleRoIAlign

from .utils import AnchorGenerator, MultiScaleAnchorGenerator
from .rpn import RPNHead, RegionProposalNetwork
from .pooler import RoIAlign
from .roi_heads import RoIHeads
from .transform import Transformer

from models.backbones import resnet

class FasterRCNN(nn.Module):
    """
    Implements Faster R-CNN.

    The input image to the model is expected to be a tensor, shape [C, H, W], and should be in 0-1 range.

    The behavior of the model changes depending if it is in training or evaluation mode.

    During training, the model expects both the input tensor, as well as a target (dictionary),
    containing:
        - boxes (FloatTensor[N, 4]): the ground-truth boxes in [xmin, ymin, xmax, ymax] format, with values
          between 0-H and 0-W
        - labels (Int64Tensor[N]): the class label for each ground-truth box

    The model returns a Dict[Tensor], containing the classification and regression losses 
    for both the RPN and the R-CNN.

    During inference, the model requires only the input tensor, and returns the post-processed
    predictions as a Dict[Tensor]. The fields of the Dict are as
    follows:
        - boxes (FloatTensor[N, 4]): the predicted boxes in [xmin, ymin, xmax, ymax] format, 
          with values between 0-H and 0-W
        - labels (Int64Tensor[N]): the predicted labels
        - scores (FloatTensor[N]): the scores for each prediction
        
    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
        num_classes (int): number of output classes of the model (including the background).
        
        rpn_fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training of the RPN.
        rpn_bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training of the RPN.
        rpn_num_samples (int): number of anchors that are sampled during training of the RPN
            for computing the loss
        rpn_positive_fraction (float): proportion of positive anchors during training of the RPN
        rpn_reg_weights (Tuple[float, float, float, float]): weights for the encoding/decoding of the
            bounding boxes
        rpn_pre_nms_top_n_train (int): number of proposals to keep before applying NMS during training
        rpn_pre_nms_top_n_test (int): number of proposals to keep before applying NMS during testing
        rpn_post_nms_top_n_train (int): number of proposals to keep after applying NMS during training
        rpn_post_nms_top_n_test (int): number of proposals to keep after applying NMS during testing
        rpn_nms_thresh (float): NMS threshold used for postprocessing the RPN proposals
        
        box_fg_iou_thresh (float): minimum IoU between the proposals and the GT box so that they can be
            considered as positive during training of the classification head
        box_bg_iou_thresh (float): maximum IoU between the proposals and the GT box so that they can be
            considered as negative during training of the classification head
        box_num_samples (int): number of proposals that are sampled during training of the
            classification head
        box_positive_fraction (float): proportion of positive proposals during training of the 
            classification head
        box_reg_weights (Tuple[float, float, float, float]): weights for the encoding/decoding of the
            bounding boxes
        box_score_thresh (float): during inference, only return proposals with a classification score
            greater than box_score_thresh
        box_nms_thresh (float): NMS threshold for the prediction head. Used during inference
        box_num_detections (int): maximum number of detections, for all classes.
        
    """
    
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
                 box_score_thresh=0.1, box_nms_thresh=0.2, box_num_detections=300):
        super().__init__()
        self.backbone = backbone
        out_channels = backbone.out_channels
 
        #------------- RPN --------------------------
        # anchor_sizes = (128, 256, 512)
        anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
        anchor_ratios = (0.5, 1, 2)
        # num_anchors = len(anchor_sizes) * len(anchor_ratios)
        num_anchors = len(anchor_ratios)
        # rpn_anchor_generator = AnchorGenerator(anchor_sizes, anchor_ratios)
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
        # box_roi_pool = RoIAlign(output_size=(7, 7), sampling_ratio=2)
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
        
        #------------ Transformer --------------------------
        self.transformer = Transformer(
            min_size=400, max_size=800, 
            image_mean=[0.485, 0.456, 0.406], 
            image_std=[0.229, 0.224, 0.225],
            resize_to=(400, 400))

        self.transformer_both = Transformer(
            min_size=400, max_size=400, 
            image_mean=[0.485, 0.456, 0.406, 0.0], 
            image_std=[0.229, 0.224, 0.225, 1.0],
            resize_to=(400,400))
        
    def forward(self, images, targets=None):
        if not isinstance(images, (list, tuple)):
            images = [images]
        
        original_image_sizes = []
        for img in images:
            val = img.shape[-2:]
            original_image_sizes.append((val[0], val[1]))

        if images[0].shape[0] == 4: 
            images, targets = self.transformer_both(images, targets)
        else:
            images, targets = self.transformer(images, targets)
        
        if isinstance(images, (list, tuple)):
            images = torch.stack(images)

        features = self.backbone(images)
        
        if isinstance(features, torch.Tensor):
            features = {'0': features}
        
        # Get image shapes for each image in the batch
        image_shapes = [(img.shape[-2], img.shape[-1]) for img in images]
        
        proposals, rpn_losses = self.rpn(features, image_shapes, targets)
        detections, detector_losses = self.head(features, proposals, image_shapes, targets)
        
        detections = self.transformer.postprocess(detections, image_shapes, original_image_sizes)
        
        losses = {}
        if self.training:
            losses.update(rpn_losses)
            losses.update(detector_losses)
            return losses
            
        return detections
        
        
class FastRCNNPredictor(nn.Module):
    def __init__(self, in_channels, mid_channels, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, mid_channels)
        self.fc2 = nn.Linear(mid_channels, mid_channels)
        self.cls_score = nn.Linear(mid_channels, num_classes)
        self.bbox_pred = nn.Linear(mid_channels, num_classes * 4)
        
    def forward(self, x):
        if isinstance(x, (list, tuple)):
            x = torch.cat(x, dim=0)
            
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        score = self.cls_score(x)
        bbox_delta = self.bbox_pred(x)
 
        return score, bbox_delta
    
    
class MaskRCNNPredictor(nn.Sequential):
    def __init__(self, in_channels, layers, dim_reduced, num_classes):
        """
        Arguments:
            in_channels (int)
            layers (Tuple[int])
            dim_reduced (int)
            num_classes (int)
        """
        
        d = OrderedDict()
        next_feature = in_channels
        for layer_idx, layer_features in enumerate(layers, 1):
            d['mask_fcn{}'.format(layer_idx)] = nn.Conv2d(next_feature, layer_features, 3, 1, 1)
            d['relu{}'.format(layer_idx)] = nn.ReLU(inplace=True)
            next_feature = layer_features
        
        d['mask_conv5'] = nn.ConvTranspose2d(next_feature, dim_reduced, 2, 2, 0)
        d['relu5'] = nn.ReLU(inplace=True)
        d['mask_fcn_logits'] = nn.Conv2d(dim_reduced, num_classes, 1, 1, 0)
        super().__init__(d)

        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
                
    
class ResBackbone(nn.Module):
    def __init__(self, backbone_name, pretrained, in_channels=3):
        super().__init__()
        body = models.resnet.__dict__[backbone_name](
            pretrained=pretrained, norm_layer=misc.FrozenBatchNorm2d)

        if in_channels == 4:
            old_conv = body.conv1
            body.conv1 = torch.nn.Conv2d(
                4, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                stride=old_conv.stride, padding=old_conv.padding, bias=old_conv.bias is not None
            )
            with torch.no_grad(): 
                body.conv1.weight[:, :3] = old_conv.weight
                # Initialize the 4th channel as the mean of the first 3
                body.conv1.weight[:, 3] = old_conv.weight.mean(dim=1)
            print("ResNet conv1 patched for 4-channel input")
        
        for name, parameter in body.named_parameters():
            if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
                
        self.body = nn.ModuleDict(d for i, d in enumerate(body.named_children()) if i < 8)
        in_channels = 2048
        self.out_channels = 256
        
        # ----- New implementation with FPN ----- #

        self.body = nn.Sequential(
            OrderedDict([
                ("conv1", body.conv1),
                ("bn1", body.bn1),
                ("relu", body.relu),
                ("maxpool", body.maxpool),
                ("layer1", body.layer1),
                ("layer2", body.layer2),
                ("layer3", body.layer3),
                ("layer4", body.layer4),
            ])
        )

        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[256, 512, 1024, 2048],
            out_channels=self.out_channels,
            extra_blocks=LastLevelMaxPool()
        )
        
        # --------------------------------------- #
        
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        # for module in self.body.values():
        #     x = module(x)
        # x = self.inner_block_module(x)
        # x = self.layer_block_module(x)
        # return x

        # ----- new implementation with FPN ----- #
        features = {}
        x = self.body.conv1(x)
        x = self.body.bn1(x)
        x = self.body.relu(x)
        x = self.body.maxpool(x)

        x = self.body.layer1(x)
        features["c2"] = x  # Low-level features

        x = self.body.layer2(x)
        features["c3"] = x

        x = self.body.layer3(x)
        features["c4"] = x

        x = self.body.layer4(x)
        features["c5"] = x  # Highest-level features

        return self.fpn(features)  # Returns P2, P3, P4, P5, P6

    
def fasterrcnn_resnet50(pretrained, num_classes, pretrained_backbone=True, args=None):
    """
    Constructs a Faster R-CNN model with a ResNet-50 backbone.
    
    Arguments:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017.
        num_classes (int): number of classes (including the background).
    """
    
    if pretrained:
        backbone_pretrained = False
        
    backbone = ResBackbone('resnet50', pretrained_backbone)
    # backbone = getattr(resnet, 'resnet50')(pretrained=pretrained, reduction=False)

    model = FasterRCNN(backbone, num_classes)
    
    if pretrained:
        model_urls = {
            'maskrcnn_resnet50_fpn_coco':
                'https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth',
            'fasterrcnn_resnet50_fpn_coco':
                'https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth'
        }
        model_state_dict = load_url(model_urls['fasterrcnn_resnet50_fpn_coco'])
        model_keys = list(model_state_dict.keys())
        print(model_keys)
        
        pretrained_msd = list(model_state_dict.values())
        del_list = [i for i in range(265, 271)] + [i for i in range(273, 279)]
        for i, del_idx in enumerate(del_list):
            pretrained_msd.pop(del_idx - i)

        msd = model.state_dict()
        skip_list = [271, 272, 273, 274, 279, 280, 281, 282, 293, 294]
        if num_classes == 91:
            skip_list = [271, 272, 273, 274]
        for i, name in enumerate(msd):
            if i in skip_list or not name.startswith("backbone.body"):
                print(f"skipped: {name} - {model_keys[i]}")
                continue
            print(f"assigned: {name} - {model_keys[i]}")
            msd[name].copy_(pretrained_msd[i])

        msd["backbone.inner_block_module.weight"].copy_(model_state_dict["backbone.fpn.inner_blocks.3.weight"])
        msd["backbone.inner_block_module.bias"].copy_(model_state_dict["backbone.fpn.inner_blocks.3.bias"])
        msd["backbone.layer_block_module.weight"].copy_(model_state_dict["backbone.fpn.layer_blocks.3.weight"])
        msd["backbone.layer_block_module.bias"].copy_(model_state_dict["backbone.fpn.layer_blocks.3.bias"])
        msd["rpn.head.conv.weight"].copy_(model_state_dict["rpn.head.conv.weight"])
        msd["rpn.head.conv.bias"].copy_(model_state_dict["rpn.head.conv.bias"])
        msd['head.box_predictor.fc1.weight'].copy_(model_state_dict['roi_heads.box_head.fc6.weight'])
        msd['head.box_predictor.fc1.bias'].copy_(model_state_dict['roi_heads.box_head.fc6.bias'])
        msd['head.box_predictor.fc2.weight'].copy_(model_state_dict['roi_heads.box_head.fc7.weight'])
        msd['head.box_predictor.fc2.bias'].copy_(model_state_dict['roi_heads.box_head.fc7.bias'])

            
        model.load_state_dict(msd)
    
    return model

def fasterrcnn_fpn_resnet50(pretrained, num_classes, pretrained_backbone=True, args=None):
    """
    Constructs a Faster R-CNN model with a ResNet-50 backbone.
    
    Arguments:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017.
        num_classes (int): number of classes (including the background).
    """
    
    if pretrained:
        backbone_pretrained = False
        
    backbone = ResBackbone('resnet50', pretrained_backbone)

    model = FasterRCNN(backbone, num_classes)
    
    # skip_first_conv = True if args.frcnn_input == 'both' else False

    if pretrained:
        model_urls = {
            'fasterrcnn_resnet50_fpn_coco':
                'https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth'
        }
        model_state_dict = load_url(model_urls['fasterrcnn_resnet50_fpn_coco'])
        
        pretrained_msd = list(model_state_dict.values())

        msd = model.state_dict()
        skip_list = [291, 292, 293, 294]
        if num_classes == 91:
            skip_list = [271, 272, 273, 274]
        for i, name in enumerate(msd):
            if i in skip_list:
                continue

            # if skip_first_conv and ("backbone.body.conv1.weight" in name or "backbone.body.conv1.bias" in name):
            #     print(f"Skipped (4ch): {name}")
            #     continue

            msd[name].copy_(pretrained_msd[i])

            
        model.load_state_dict(msd)
    
    return model