import torch
import torch.nn.functional as F
from torch import nn

from .box_ops import BoxCoder, box_iou, process_box, nms
from .utils import Matcher, BalancedPositiveNegativeSampler

class RPNHead(nn.Module):
    def __init__(self, in_channels, num_anchors):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, 1)
        self.bbox_pred = nn.Conv2d(in_channels, 4 * num_anchors, 1)
        
        for l in self.children():
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)
            
    def forward(self, x):
        x = F.relu(self.conv(x))
        logits = self.cls_logits(x)
        bbox_reg = self.bbox_pred(x)
        return logits, bbox_reg
    

class RegionProposalNetwork(nn.Module):
    def __init__(self, anchor_generator, head, 
                 fg_iou_thresh, bg_iou_thresh,
                 num_samples, positive_fraction,
                 reg_weights,
                 pre_nms_top_n, post_nms_top_n, nms_thresh):
        super().__init__()
        
        self.anchor_generator = anchor_generator
        self.head = head
        
        self.proposal_matcher = Matcher(fg_iou_thresh, bg_iou_thresh, allow_low_quality_matches=True)
        self.fg_bg_sampler = BalancedPositiveNegativeSampler(num_samples, positive_fraction)
        self.box_coder = BoxCoder(reg_weights)
        
        self._pre_nms_top_n = pre_nms_top_n
        self._post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.min_size = 1
                
    def create_proposal(self, anchor, objectness, pred_bbox_delta, image_shape):
        if self.training:
            pre_nms_top_n = self._pre_nms_top_n['training']
            post_nms_top_n = self._post_nms_top_n['training']
        else:
            pre_nms_top_n = self._pre_nms_top_n['testing']
            post_nms_top_n = self._post_nms_top_n['testing']
            
        pre_nms_top_n = min(objectness.shape[0], pre_nms_top_n)
        top_n_idx = objectness.topk(pre_nms_top_n)[1]
        score = objectness[top_n_idx]
        proposal = self.box_coder.decode(pred_bbox_delta[top_n_idx], anchor[top_n_idx])
        
        proposal, score = process_box(proposal, score, image_shape, self.min_size)
        keep = nms(proposal, score, self.nms_thresh)[:post_nms_top_n] 
        proposal = proposal[keep]
        return proposal
    
    def compute_loss(self, objectness, pred_bbox_delta, gt_box, anchor):
        iou = box_iou(gt_box, anchor)
        label, matched_idx = self.proposal_matcher(iou)
        
        pos_idx, neg_idx = self.fg_bg_sampler(label)
        idx = torch.cat((pos_idx, neg_idx))
        regression_target = self.box_coder.encode(gt_box[matched_idx[pos_idx]], anchor[pos_idx])
        
        objectness_loss = F.binary_cross_entropy_with_logits(objectness[idx], label[idx])
        box_loss = F.l1_loss(pred_bbox_delta[pos_idx], regression_target, reduction='sum') / idx.numel()

        return objectness_loss, box_loss
        
    def forward(self, features, image_shapes, targets=None):
        """
        Arguments:
            features (Dict[str, Tensor]): FPN feature maps for each level
            image_shapes (List[Tuple[int, int]]): List of (h, w) for each image
            targets (List[Dict], optional): List of target dictionaries for each image
        """
        # Generate anchors for each FPN level and each image
        batch_size = len(image_shapes)
        all_anchors = []
        all_num_anchors_per_level = []

        for i in range(batch_size):
            anchors, num_anchors = self.anchor_generator(features, image_shapes[i])
            all_anchors.append(anchors)
            all_num_anchors_per_level.append(num_anchors)

        all_objectness = []
        all_pred_bbox_deltas = []
        all_proposals = []

        # Process each image in the batch
        for i in range(batch_size):
            image_objectness = []
            image_pred_bbox_deltas = []
            start_idx = 0

            # Process each feature level for this image
            for feature_name, num_anchors in zip(features.keys(), all_num_anchors_per_level[i]):
                feature = features[feature_name][i:i+1]  # Get feature map for this image
                anchors = all_anchors[i][start_idx:start_idx + num_anchors]
                start_idx += num_anchors

                # RPN head forward pass
                objectness, pred_bbox_delta = self.head(feature)

                # Reshape predictions
                objectness = objectness.permute(0, 2, 3, 1).flatten()
                pred_bbox_delta = pred_bbox_delta.permute(0, 2, 3, 1).reshape(-1, 4)

                image_objectness.append(objectness)
                image_pred_bbox_deltas.append(pred_bbox_delta)

            # Concatenate predictions from all levels for this image
            image_objectness = torch.cat(image_objectness, dim=0)
            image_pred_bbox_deltas = torch.cat(image_pred_bbox_deltas, dim=0)

            # Generate proposals for this image
            proposals = self.create_proposal(
                all_anchors[i],
                image_objectness.detach(),
                image_pred_bbox_deltas.detach(),
                image_shapes[i]
            )

            all_objectness.append(image_objectness)
            all_pred_bbox_deltas.append(image_pred_bbox_deltas)
            all_proposals.append(proposals)

        if self.training:
            assert targets is not None
            rpn_losses = []
            
            # Compute loss for each image in the batch
            for i in range(batch_size):
                objectness_loss, box_loss = self.compute_loss(
                    all_objectness[i],
                    all_pred_bbox_deltas[i],
                    targets[i]['boxes'],
                    all_anchors[i]
                )
                rpn_losses.append({
                    'rpn_objectness_loss': objectness_loss,
                    'rpn_box_loss': box_loss
                })

            # Average losses across the batch
            avg_losses = {
                'rpn_objectness_loss': torch.mean(torch.stack([l['rpn_objectness_loss'] for l in rpn_losses])),
                'rpn_box_loss': torch.mean(torch.stack([l['rpn_box_loss'] for l in rpn_losses]))
            }
            return all_proposals, avg_losses

        return all_proposals, {}

class RegionProposalNetworkSingleFeature(nn.Module):
    def __init__(self, anchor_generator, head, 
                 fg_iou_thresh, bg_iou_thresh,
                 num_samples, positive_fraction,
                 reg_weights,
                 pre_nms_top_n, post_nms_top_n, nms_thresh):
        super().__init__()
        
        self.anchor_generator = anchor_generator
        self.head = head
        
        self.proposal_matcher = Matcher(fg_iou_thresh, bg_iou_thresh, allow_low_quality_matches=True)
        self.fg_bg_sampler = BalancedPositiveNegativeSampler(num_samples, positive_fraction)
        self.box_coder = BoxCoder(reg_weights)
        
        self._pre_nms_top_n = pre_nms_top_n
        self._post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.min_size = 1
                
    def create_proposal(self, anchor, objectness, pred_bbox_delta, image_shape):
        if self.training:
            pre_nms_top_n = self._pre_nms_top_n['training']
            post_nms_top_n = self._post_nms_top_n['training']
        else:
            pre_nms_top_n = self._pre_nms_top_n['testing']
            post_nms_top_n = self._post_nms_top_n['testing']
            
        pre_nms_top_n = min(objectness.shape[0], pre_nms_top_n)
        top_n_idx = objectness.topk(pre_nms_top_n)[1]
        score = objectness[top_n_idx]
        proposal = self.box_coder.decode(pred_bbox_delta[top_n_idx], anchor[top_n_idx])
        
        proposal, score = process_box(proposal, score, image_shape, self.min_size)
        keep = nms(proposal, score, self.nms_thresh)[:post_nms_top_n] 
        proposal = proposal[keep]
        return proposal
    
    def compute_loss(self, objectness, pred_bbox_delta, gt_box, anchor):
        iou = box_iou(gt_box, anchor)
        label, matched_idx = self.proposal_matcher(iou)
        
        pos_idx, neg_idx = self.fg_bg_sampler(label)
        idx = torch.cat((pos_idx, neg_idx))
        regression_target = self.box_coder.encode(gt_box[matched_idx[pos_idx]], anchor[pos_idx])
        
        objectness_loss = F.binary_cross_entropy_with_logits(objectness[idx], label[idx])
        box_loss = F.l1_loss(pred_bbox_delta[pos_idx], regression_target, reduction='sum') / idx.numel()

        return objectness_loss, box_loss
        
    def forward(self, features, image_shapes, targets=None):
        """
        Arguments:
            features (Tensor): [B, C, H, W] feature map for each image
            image_shapes (List[Tuple[int, int]]): List of (h, w) for each image
            targets (List[Dict], optional): List of target dictionaries for each image
        """
        batch_size = len(image_shapes)
        all_anchors = []
        all_objectness = []
        all_pred_bbox_deltas = []
        all_proposals = []

        # Generate anchors and predictions for each image in the batch
        for i in range(batch_size):
            feature = features[i:i+1]  # [1, C, H, W]
            anchors= self.anchor_generator(feature, image_shapes[i])
            objectness, pred_bbox_delta = self.head(feature)
            objectness = objectness.permute(0, 2, 3, 1).flatten()
            pred_bbox_delta = pred_bbox_delta.permute(0, 2, 3, 1).reshape(-1, 4)

            # Generate proposals for this image
            proposals = self.create_proposal(
                anchors,
                objectness.detach(),
                pred_bbox_delta.detach(),
                image_shapes[i]
            )

            all_anchors.append(anchors)
            all_objectness.append(objectness)
            all_pred_bbox_deltas.append(pred_bbox_delta)
            all_proposals.append(proposals)

        if self.training:
            assert targets is not None
            rpn_losses = []
            for i in range(batch_size):
                objectness_loss, box_loss = self.compute_loss(
                    all_objectness[i],
                    all_pred_bbox_deltas[i],
                    targets[i]['boxes'],
                    all_anchors[i]
                )
                rpn_losses.append({
                    'rpn_objectness_loss': objectness_loss,
                    'rpn_box_loss': box_loss
                })
            avg_losses = {
                'rpn_objectness_loss': torch.mean(torch.stack([l['rpn_objectness_loss'] for l in rpn_losses])),
                'rpn_box_loss': torch.mean(torch.stack([l['rpn_box_loss'] for l in rpn_losses]))
            }
            return all_proposals, avg_losses

        return all_proposals, {}