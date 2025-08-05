import torch
import torch.nn as nn
from .backbone import Backbone, Neck, Conv, C3k2, A2C2f, FusionBackbone
from .detect import Detect
from .modules import Concat
from .losses import YoloV12Loss, YOLOLoss
from models.rcnn.transform import Transformer

class YOLOv12(nn.Module):
    """Complete YOLOv12 Model"""
    def __init__(self, num_classes=80, depth_multiple=0.50, width_multiple=0.25, max_channels=1024, pretrained=True, path='yolo12n.pt'):
        super().__init__()
        self.backbone = Backbone(width_multiple=width_multiple, depth_multiple=depth_multiple, max_channels=max_channels)
        self.neck = Neck(width_multiple=width_multiple, depth_multiple=depth_multiple, max_channels=max_channels)
        self.head = Detect(
            ch=[
                min(int(256 * width_multiple), max_channels), 
                min(int(512 * width_multiple), max_channels), 
                min(int(1024 * width_multiple), max_channels)
                ], 
            num_classes=num_classes
        )
        self.loss_fn = YoloV12Loss(num_classes=num_classes)
        self.yolo_loss = YOLOLoss(model=self, tal_topk=10)
        self.num_classes = num_classes

        self.transform = Transformer( 
            min_size=640, max_size=640, 
            image_mean=[0.485, 0.456, 0.406], 
            image_std=[0.229, 0.224, 0.225],
            resize_to=(640, 640)
        )

        self.chm_transform = Transformer( 
            min_size=640, max_size=640, 
            image_mean=[0.0, 0.0, 0.0], 
            image_std=[1.0, 1.0, 1.0],
            resize_to=(640, 640)
        )

        if pretrained:
            self.load_pretrained(path=path)

    def load_pretrained(self, path='yolo12n.pt'):
        model_state_dict = self.state_dict()
         
        ckpt = torch.load(path, map_location="cpu")['model'].model
        ckpt_state_dict = ckpt.state_dict()

        mapping = {
            '0': 'backbone.stem',
            '1': 'backbone.stage1.0',
            '2': 'backbone.stage1.1',
            '3': 'backbone.stage2.0',
            '4': 'backbone.stage2.1',
            '5': 'backbone.stage3.0',
            '6': 'backbone.stage3.1',
            '7': 'backbone.stage4.0',
            '8': 'backbone.stage4.1',
            '9': 'neck.upsample1',
            '10': 'neck.concat1',
            '11': 'neck.merge1',
            '12': 'neck.upsample2',
            '13': 'neck.concat2',
            '14': 'neck.merge2',
            '15': 'neck.downsample1',
            '16': 'neck.concat3',
            '17': 'neck.merge3',
            '18': 'neck.downsample2',
            '19': 'neck.concat4',
            '20': 'neck.merge4',
            '21': 'head'
        }

        for i, name in enumerate(ckpt_state_dict):
            for key, value in mapping.items():
                if name.startswith(key):
                    pretrained_name = name.replace(key, value)
                    if pretrained_name in model_state_dict:
                        if model_state_dict[pretrained_name].shape == ckpt_state_dict[name].shape:
                            model_state_dict[pretrained_name].copy_(ckpt_state_dict[name]) 

        self.load_state_dict(model_state_dict, strict=True)

    def forward(self, rgb_images, targets=None): 
        original_image_sizes = []
        for img in rgb_images:
            val = img.shape[-2:]
            original_image_sizes.append((val[0], val[1]))

        rgb_images, targets = self.transform(rgb_images, targets)

        p3, p4, p5 = self.backbone(rgb_images)
        p3, p4, p5 = self.neck(p3, p4, p5)

        if self.training:
            x = self.head([p3, p4, p5])
            self.yolo_loss = YOLOLoss(model=self, tal_topk=10)
            loss = self.yolo_loss(x, targets)

            return loss

        predictions, pred_dist, x = self.head([p3, p4, p5])
        batch_size = predictions.shape[0]
        num_preds = predictions.shape[2]
        predictions = predictions.permute(0, 2, 1).contiguous()
        pred_dist = pred_dist.permute(0, 2, 1).contiguous() 

        boxes = predictions[..., :4]
        scores = predictions[..., 4:4 + self.num_classes]

        output = []
        for i in range(batch_size):
            output.append({
                'boxes': boxes[i],
                'scores': scores[i],
                'pred_dist': pred_dist[i]
            })

        output = self.post_process(output, original_image_sizes) 
        
        return output

    @staticmethod
    def post_process(results, original_sizes):
        """Post-process the model outputs"""
        post_processed_results = []
        for i, result in enumerate(results):
            # Get classification scores
            scores = torch.sigmoid(result['scores'])
            
            # Convert boxes from (cx, cy, w, h) to (x1, y1, x2, y2)
            # boxes = result['boxes'].clamp(0, 1)
            boxes = result['boxes']
            
            # Scale boxes to original image size
            h, w = original_sizes[i]
            boxes = boxes * torch.tensor([w, h, w, h], dtype=torch.float32, device=boxes.device) / torch.tensor([640, 640, 640, 640], dtype=torch.float32, device=boxes.device)
            
            post_processed_results.append({
                'scores': scores,
                'boxes': boxes,
            })
        
        return post_processed_results

class FusionYolov12(YOLOv12):
    def __init__(self, num_classes=80, depth_multiple=0.50, width_multiple=0.25, max_channels=1024, pretrained=True, path='yolo12n.pt'):
        super().__init__(num_classes=num_classes, depth_multiple=depth_multiple, width_multiple=width_multiple, max_channels=max_channels, pretrained=pretrained, path=path)
        
        self.backbone = FusionBackbone(width_multiple=width_multiple, depth_multiple=depth_multiple, max_channels=max_channels)

        if pretrained:
            self.load_pretrained(path=path)
        
    def load_pretrained(self, path='yolo12n.pt'):
        model_state_dict = self.state_dict()
         
        ckpt = torch.load(path, map_location="cpu")['model'].model
        ckpt_state_dict = ckpt.state_dict()

        mapping = {
            'backbone.rgb_stem': '0',
            'backbone.rgb_stage1.0': '1',
            'backbone.rgb_stage1.1': '2',
            'backbone.rgb_stage2.0': '3',
            'backbone.rgb_stage2.1': '4',
            'backbone.rgb_stage3.0': '5',
            'backbone.rgb_stage3.1': '6',
            'backbone.rgb_stage4.0': '7',
            'backbone.rgb_stage4.1': '8',
            'backbone.chm_stem': '0',
            'backbone.chm_stage1.0': '1',
            'backbone.chm_stage1.1': '2',
            'backbone.chm_stage2.0': '3',
            'backbone.chm_stage2.1': '4',
            'backbone.chm_stage3.0': '5',
            'backbone.chm_stage3.1': '6',
            'backbone.chm_stage4.0': '7',
            'backbone.chm_stage4.1': '8',
            'neck.upsample1': '9',
            'neck.concat1': '10',
            'neck.merge1': '11',
            'neck.upsample2': '12',
            'neck.concat2': '13',
            'neck.merge2': '14',
            'neck.downsample1': '15',
            'neck.concat3': '16',
            'neck.merge3': '17',
            'neck.downsample2': '18',
            'neck.concat4': '19',
            'neck.merge4': '20',
            'head': '21'
        }

        for i, name in enumerate(model_state_dict):
            for key, value in mapping.items():
                if name.startswith(key):
                    pretrained_name = name.replace(key, value)
                    if pretrained_name in ckpt_state_dict:
                        if model_state_dict[name].shape == ckpt_state_dict[pretrained_name].shape:
                            model_state_dict[name].copy_(ckpt_state_dict[pretrained_name])

        self.load_state_dict(model_state_dict, strict=True)

    def forward(self, rgb_images, chm_images, targets=None):
        original_image_sizes = []
        for img in rgb_images:
            val = img.shape[-2:]
            original_image_sizes.append((val[0], val[1]))
            
        rgb_images, targets = self.transform(rgb_images, targets)
        chm_images, _ = self.chm_transform(chm_images, None)

        p3, p4, p5 = self.backbone(rgb_images, chm_images)
        p3, p4, p5 = self.neck(p3, p4, p5)

        if self.training:
            x = self.head([p3, p4, p5])
            self.yolo_loss = YOLOLoss(model=self, tal_topk=10)
            loss = self.yolo_loss(x, targets)

            return loss

        predictions, pred_dist, x = self.head([p3, p4, p5])
        batch_size = predictions.shape[0]
        num_preds = predictions.shape[2]
        predictions = predictions.permute(0, 2, 1).contiguous()
        pred_dist = pred_dist.permute(0, 2, 1).contiguous() 

        boxes = predictions[..., :4]
        scores = predictions[..., 4:4 + self.num_classes]

        output = []
        for i in range(batch_size):
            output.append({
                'boxes': boxes[i],
                'scores': scores[i],
                'pred_dist': pred_dist[i]
            })

        output = self.post_process(output, original_image_sizes) 
        
        return output
        