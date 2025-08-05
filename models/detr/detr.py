from abc import abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import ResNet50_Weights
from torchvision.models._utils import IntermediateLayerGetter
import math

import utils.misc
from .backbone import Backbone, CBAM_Backbone
from .transformer import Transformer
from .position_encoding import PositionEmbeddingSine
from .backbone import Joiner, FusionJoiner
from .box_ops import box_cxcywh_to_xyxy
from models.rcnn import transform
import utils

class DETR(nn.Module):
    def __init__(self, num_classes, num_queries=100, backbone_name='resnet50', pretrained=True):
        """
        Simplified DETR implementation
        Args:
            num_classes (int): Number of object classes
            num_queries (int): Number of object queries
            backbone_name (str): Name of the backbone to use ('resnet50', 'resnet101', etc.)
            pretrained (bool): Whether to use pretrained backbone
        """
        super().__init__()
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim = 256
        
        # Build backbone
        self.backbone = self._build_backbone(backbone_name, self.hidden_dim)
        
        # Build transformer
        self.transformer = Transformer(
            d_model=hidden_dim,
            return_intermediate_dec=True
        )
        
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        self.input_proj = nn.Conv2d(self.backbone.num_channels, hidden_dim, kernel_size=1)
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)  # +1 for no-object class
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        
        # Position encoding
        self.pos_encoder = PositionEmbeddingSine(hidden_dim // 2)

        self.transform = transform.Transformer(
            min_size=480, max_size=800, 
            image_mean=[0.485, 0.456, 0.406], 
            image_std=[0.229, 0.224, 0.225]
        )
    
    def _build_backbone(self, backbone_name, hidden_dim):
        backbone = Backbone(backbone_name, train_backbone=True, return_interm_layers=False, dilation=False)
        pos_enc = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
        backbone_with_pos_enc = Joiner(backbone, pos_enc)
        backbone_with_pos_enc.num_channels = backbone.num_channels

        return backbone_with_pos_enc
    
    def forward(self, images):
        """Forward pass of the model"""
        if not isinstance(images, list):
            images = [images]

        original_sizes = [img.shape[-2:] for img in images]

        images, _ = self.transform(images)

        if isinstance(images, (list, tuple)):
            images = torch.stack(images)
        tensor_list = utils.misc.nested_tensor_from_tensor_list(images)
        tensor_list = tensor_list.to(images.device)

        features, pos = self.backbone(tensor_list)

        src, mask = features[-1].decompose()
        assert mask is not None
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]
        
        # Predict classes and boxes
        outputs_class = self.class_embed(hs)
        outputs_coord = torch.sigmoid(self.bbox_embed(hs))

        results = {
            'scores': outputs_class[-1],
            'boxes': outputs_coord[-1]
        }

        if self.training:
            return results
        
        return self.post_process(results, original_sizes)

    @staticmethod
    def post_process(results, original_sizes):
        """Post-process the model outputs"""
        post_processed_results = []
        for i, (pred_logits, pred_boxes) in enumerate(zip(results['scores'], results['boxes'])):
            # Get classification scores
            scores = F.softmax(pred_logits, dim=-1)
            
            # Convert boxes from (cx, cy, w, h) to (x1, y1, x2, y2)
            boxes = box_cxcywh_to_xyxy(pred_boxes)
            boxes = boxes.clamp(0, 1)
            
            # Scale boxes to original image size
            h, w = original_sizes[i]
            boxes = boxes * torch.tensor([w, h, w, h], dtype=torch.float32, device=boxes.device)
            
            post_processed_results.append({
                'scores': scores,
                'boxes': boxes,
            })
        
        return post_processed_results
    
    @staticmethod
    def box_cxcywh_to_xyxy(x):
        """Convert bbox coordinates from (cx, cy, w, h) to (x1, y1, x2, y2)"""
        x_c, y_c, w, h = x.unbind(-1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
             (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=-1)

class CBAMDETR(DETR):
    def __init__(self, num_classes, num_queries=100, backbone_name='resnet50', pretrained=True):
        super().__init__(num_classes, num_queries, backbone_name, pretrained)
    
    def _build_backbone(self, backbone_name, hidden_dim):
        backbone = CBAM_Backbone(backbone_name, train_backbone=True, return_interm_layers=False, dilation=False)
        
        pos_enc = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
        backbone_with_pos_enc = FusionJoiner(backbone, pos_enc)
        backbone_with_pos_enc.num_channels = backbone.num_channels

        return backbone_with_pos_enc

    def forward(self, rgb_imgs, chm_imgs):
        if not isinstance(rgb_imgs, list):
            rgb_imgs = [rgb_imgs]
        if not isinstance(chm_imgs, list):
            chm_imgs = [chm_imgs] 
            
        original_sizes = [img.shape[-2:] for img in rgb_imgs]
        rgb_imgs, _ = self.transform(rgb_imgs)
        chm_imgs, _ = self.transform(chm_imgs)
        
        rgb = utils.misc.nested_tensor_from_tensor_list(rgb_imgs)
        chm = utils.misc.nested_tensor_from_tensor_list(chm_imgs)
        
        features, pos = self.backbone(rgb, chm)

        src, mask = features[-1].decompose()
        assert mask is not None
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]
        
        # Predict classes and boxes
        outputs_class = self.class_embed(hs)
        outputs_coord = torch.sigmoid(self.bbox_embed(hs))

        results = {
            'scores': outputs_class[-1],
            'boxes': outputs_coord[-1]
        }

        if self.training:
            return results
        
        return self.post_process(results, original_sizes)

class MLP(nn.Module):
    """Simple multi-layer perceptron"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        
        # Initialize weights and biases
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Apply ReLU to all but the last layer
            if i < self.num_layers - 1:
                x = F.relu(x)
            
        
        return x

def build_detr(num_classes, backbone='resnet50', num_queries=100, pretrained=True):
    """Helper function to build DETR model with specified configuration"""
    model = DETR(
        num_classes=num_classes,
        num_queries=num_queries,
        backbone_name=backbone,
        pretrained=pretrained
    )
    
    if pretrained:
        # Load DETR pretrained weights
        checkpoint = torch.hub.load_state_dict_from_url(url='https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth', map_location='cpu', check_hash=True)
 
        # Remove the class_embed layer from the checkpoint
        del checkpoint["model"]["class_embed.weight"]
        del checkpoint["model"]["class_embed.bias"]

        model.load_state_dict(checkpoint["model"], strict=False)
    
    return model

def build_cbam_detr(num_classes, backbone='resnet50', num_queries=100, pretrained=True):
    model = CBAMDETR(
        num_classes=num_classes,
        num_queries=num_queries,
        backbone_name=backbone,
        pretrained=pretrained
    )

    if pretrained: 
        model_urls = {
            'detr_resnet50': 'https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth'
        }
        model_state_dict = torch.hub.load_state_dict_from_url(model_urls['detr_resnet50'], map_location='cpu', check_hash=True)

        # Remove the class_embed and bbox_embed layers from the checkpoint
        del model_state_dict["model"]["class_embed.weight"]
        del model_state_dict["model"]["class_embed.bias"]

        msd = model.state_dict()
        skip_list = ['class_embed.weight', 'class_embed.bias']

        for name in msd:
            if name.startswith('backbone.0.backbone.rgb_backbone') and name not in skip_list:
                pretrained_name = name.replace('backbone.0.backbone.rgb_backbone', 'backbone.0.body')
                if pretrained_name in model_state_dict["model"]:
                    msd[name].copy_(model_state_dict["model"][pretrained_name])

        for name in msd:
            if name.startswith('backbone.0.backbone.chm_backbone') and name not in skip_list:
                pretrained_name = name.replace('backbone.0.backbone.chm_backbone', 'backbone.0.body')
                if pretrained_name in model_state_dict["model"]:
                    msd[name].copy_(model_state_dict["model"][pretrained_name])

        # Load other weights
        for name in msd:
            if name not in skip_list and name in model_state_dict["model"]:
                msd[name].copy_(model_state_dict["model"][name])

        model.load_state_dict(msd, strict=True)

    return model