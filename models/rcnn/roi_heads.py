import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import MultiScaleRoIAlign

from .pooler import RoIAlign
from .utils import Matcher, BalancedPositiveNegativeSampler, roi_align
from .box_ops import BoxCoder, process_box, nms, box_iou

def diou_loss(pred_boxes, target_boxes):
    # pred_boxes, target_boxes: [N, 4] in (x1, y1, x2, y2)
    if pred_boxes.numel() == 0 or target_boxes.numel() == 0:
        return torch.tensor(0.0, device=pred_boxes.device, requires_grad=True)
    # Remove degenerate boxes
    w_pred = (pred_boxes[:, 2] - pred_boxes[:, 0]).clamp(min=1e-6)
    h_pred = (pred_boxes[:, 3] - pred_boxes[:, 1]).clamp(min=1e-6)
    w_gt = (target_boxes[:, 2] - target_boxes[:, 0]).clamp(min=1e-6)
    h_gt = (target_boxes[:, 3] - target_boxes[:, 1]).clamp(min=1e-6)
    pred_boxes = torch.stack([
        pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 0] + w_pred, pred_boxes[:, 1] + h_pred
    ], dim=1)
    target_boxes = torch.stack([
        target_boxes[:, 0], target_boxes[:, 1], target_boxes[:, 0] + w_gt, target_boxes[:, 1] + h_gt
    ], dim=1)
    iou = box_iou(pred_boxes, target_boxes).diag()  # [N]
    px = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2
    py = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2
    gx = (target_boxes[:, 0] + target_boxes[:, 2]) / 2
    gy = (target_boxes[:, 1] + target_boxes[:, 3]) / 2
    center_dist = (px - gx) ** 2 + (py - gy) ** 2
    x1 = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
    y1 = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
    x2 = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
    y2 = torch.max(pred_boxes[:, 3], target_boxes[:, 3])
    diag = (x2 - x1) ** 2 + (y2 - y1) ** 2 + 1e-7
    diou = iou - center_dist / diag
    loss = 1 - diou
    return loss.mean()

def ciou_loss(pred_boxes, target_boxes):
    if pred_boxes.numel() == 0 or target_boxes.numel() == 0:
        return torch.tensor(0.0, device=pred_boxes.device, requires_grad=True)
    # Remove degenerate boxes
    w_pred = (pred_boxes[:, 2] - pred_boxes[:, 0]).clamp(min=1e-6)
    h_pred = (pred_boxes[:, 3] - pred_boxes[:, 1]).clamp(min=1e-6)
    w_gt = (target_boxes[:, 2] - target_boxes[:, 0]).clamp(min=1e-6)
    h_gt = (target_boxes[:, 3] - target_boxes[:, 1]).clamp(min=1e-6)
    pred_boxes = torch.stack([
        pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 0] + w_pred, pred_boxes[:, 1] + h_pred
    ], dim=1)
    target_boxes = torch.stack([
        target_boxes[:, 0], target_boxes[:, 1], target_boxes[:, 0] + w_gt, target_boxes[:, 1] + h_gt
    ], dim=1)
    iou = box_iou(pred_boxes, target_boxes).diag()  # [N]
    px = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2
    py = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2
    gx = (target_boxes[:, 0] + target_boxes[:, 2]) / 2
    gy = (target_boxes[:, 1] + target_boxes[:, 3]) / 2
    center_dist = (px - gx) ** 2 + (py - gy) ** 2
    x1 = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
    y1 = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
    x2 = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
    y2 = torch.max(pred_boxes[:, 3], target_boxes[:, 3])
    diag = (x2 - x1) ** 2 + (y2 - y1) ** 2 + 1e-7
    v = (4 / (3.14159265 ** 2)) * (torch.atan(w_gt / (h_gt + 1e-7)) - torch.atan(w_pred / (h_pred + 1e-7))) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou + v + 1e-7)
    ciou = iou - center_dist / diag - alpha * v
    loss = 1 - ciou

    # Mask out NaNs before reduction
    valid = ~torch.isnan(loss)
    if valid.any():
        loss = loss[valid]
        return loss.mean()
    else:
        return torch.tensor(0.0, device=pred_boxes.device, requires_grad=True)

def fastrcnn_loss(class_logit, box_regression, label, regression_target, loss_cfg):
    # Classifier loss
    if loss_cfg.get('classifier_loss_type', 'cross_entropy') == 'focal':
        focal_loss = FocalLoss(alpha=loss_cfg.get('focal_alpha', 0.25), gamma=loss_cfg.get('focal_gamma', 2.0))
        classifier_loss = focal_loss(class_logit, label)
    elif loss_cfg.get('classifier_loss_type', 'cross_entropy') == 'cross_entropy':
        classifier_loss = F.cross_entropy(class_logit, label)

    # Optionally combine cross-entropy and focal loss
    if loss_cfg.get('classifier_loss_type', 'cross_entropy') == 'both':
        focal_loss = FocalLoss(alpha=loss_cfg.get('focal_alpha', 0.25), gamma=loss_cfg.get('focal_gamma', 2.0))
        fc = focal_loss(class_logit, label)
        ce = F.cross_entropy(class_logit, label)
        classifier_loss = loss_cfg.get('ce_weight', 0.8) * ce + loss_cfg.get('focal_weight', 0.2) * fc

    N, num_pos = class_logit.shape[0], regression_target.shape[0]
    box_regression = box_regression.reshape(N, -1, 4)
    box_regression, label = box_regression[:num_pos], label[:num_pos]
    box_idx = torch.arange(num_pos, device=label.device)

    pred_boxes = box_regression[box_idx, label]
    target_boxes = regression_target

    l1_weight = loss_cfg.get('l1', 0.8)
    ciou_weight = loss_cfg.get('ciou', 0.2)
    l1 = F.smooth_l1_loss(pred_boxes, target_boxes, reduction='sum') / N
    ciou = ciou_loss(pred_boxes, target_boxes)
    box_reg_loss = l1_weight * l1 + ciou_weight * ciou

    return classifier_loss, box_reg_loss

def maskrcnn_loss(mask_logit, proposal, matched_idx, label, gt_mask):
    matched_idx = matched_idx[:, None].to(proposal)
    roi = torch.cat((matched_idx, proposal), dim=1)
            
    M = mask_logit.shape[-1]
    gt_mask = gt_mask[:, None].to(roi)
    mask_target = roi_align(gt_mask, roi, 1., M, M, -1)[:, 0]

    idx = torch.arange(label.shape[0], device=label.device)
    mask_loss = F.binary_cross_entropy_with_logits(mask_logit[idx, label], mask_target)
    return mask_loss
    

class RoIHeads(nn.Module):
    def __init__(self, box_roi_pool, box_predictor,
                 fg_iou_thresh, bg_iou_thresh,
                 num_samples, positive_fraction,
                 reg_weights,
                 score_thresh, nms_thresh, num_detections):
        super().__init__()
        self.box_roi_pool = box_roi_pool
        self.box_predictor = box_predictor
        
        self.mask_roi_pool = None
        self.mask_predictor = None
        
        self.proposal_matcher = Matcher(fg_iou_thresh, bg_iou_thresh, allow_low_quality_matches=False)
        self.fg_bg_sampler = BalancedPositiveNegativeSampler(num_samples, positive_fraction)
        self.box_coder = BoxCoder(reg_weights)
        
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.num_detections = num_detections
        self.min_size = 1
        
    def has_mask(self):
        if self.mask_roi_pool is None:
            return False
        if self.mask_predictor is None:
            return False
        return True
    
    def select_training_samples(self, proposals, targets):
        batch_size = len(proposals)
        all_proposals = []
        all_labels = []
        all_regression_targets = []
        all_matched_idxs = []

        for i in range(batch_size):
            proposal = proposals[i]
            target = targets[i]
            gt_box = target['boxes']
            gt_label = target['labels']
            proposal = torch.cat((proposal, gt_box))
            
            iou = box_iou(gt_box, proposal)
            pos_neg_label, matched_idx = self.proposal_matcher(iou)
            pos_idx, neg_idx = self.fg_bg_sampler(pos_neg_label)
            idx = torch.cat((pos_idx, neg_idx))
            
            regression_target = self.box_coder.encode(gt_box[matched_idx[pos_idx]], proposal[pos_idx])
            proposal = proposal[idx]
            matched_idx = matched_idx[idx]
            label = gt_label[matched_idx]
            num_pos = pos_idx.shape[0]
            label[num_pos:] = 0

            all_proposals.append(proposal)
            all_labels.append(label)
            all_regression_targets.append(regression_target)
            all_matched_idxs.append(matched_idx)

        return all_proposals, all_matched_idxs, all_labels, all_regression_targets
    
    def fastrcnn_inference(self, class_logits, box_regressions, proposals, image_shapes):
        batch_size = len(proposals)
        result_boxes = []
        result_scores = []
        result_labels = []

        for i in range(batch_size):
            class_logit = class_logits[i]
            box_regression = box_regressions[i]
            proposal = proposals[i]
            image_shape = image_shapes[i]

            N, num_classes = class_logit.shape
            device = class_logit.device
            pred_score = F.softmax(class_logit, dim=-1)
            box_regression = box_regression.reshape(N, -1, 4)
            
            boxes = []
            scores = []
            labels = []
            for l in range(1, num_classes):
                score, box_delta = pred_score[:, l], box_regression[:, l]

                keep = score >= self.score_thresh
                box, score, box_delta = proposal[keep], score[keep], box_delta[keep]
                box = self.box_coder.decode(box_delta, box)
                
                box, score = process_box(box, score, image_shape, self.min_size)
                
                keep = nms(box, score, self.nms_thresh)[:self.num_detections]
                box, score = box[keep], score[keep]
                label = torch.full((len(keep),), l, dtype=keep.dtype, device=device)
                
                boxes.append(box)
                scores.append(score)
                labels.append(label)

            if len(boxes) > 0:
                result_boxes.append(torch.cat(boxes))
                result_scores.append(torch.cat(scores))
                result_labels.append(torch.cat(labels))
            else:
                result_boxes.append(torch.zeros((0, 4), device=device))
                result_scores.append(torch.zeros((0,), device=device))
                result_labels.append(torch.zeros((0,), device=device, dtype=torch.int64))

        return dict(boxes=result_boxes, labels=result_labels, scores=result_scores)
    
    def forward(self, features, proposals, image_shapes, targets=None):
        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)

        # Get box features for all proposals across all images
        box_features = self.box_roi_pool(features, proposals, image_shapes)
        
        # Get predictions for all proposals
        class_logits, box_regression = self.box_predictor(box_features)
        
        result = {}
        losses = {}

        if self.training:
            # Compute loss over all proposals
            classifier_loss, box_reg_loss = fastrcnn_loss(
                class_logits, box_regression, torch.cat(labels), torch.cat(regression_targets),
                loss_cfg=getattr(self, 'loss_cfg', {'l1': 1.0, 'ciou': 0.0, 'classifier_loss_type': 'cross_entropy', 'focal_alpha': 0.25, 'focal_gamma': 2.0})
            )
            losses = dict(roi_classifier_loss=classifier_loss, roi_box_loss=box_reg_loss)
        else:
            # Split predictions back into per-image results
            num_proposals_per_image = [len(p) for p in proposals]
            class_logits = class_logits.split(num_proposals_per_image, dim=0)
            box_regression = box_regression.split(num_proposals_per_image, dim=0)
            
            result = self.fastrcnn_inference(class_logits, box_regression, proposals, image_shapes)
            
        if self.has_mask():
            if self.training:
                num_pos = sum(len(target['boxes']) for target in targets)
                
                mask_proposals = []
                pos_matched_idxs = []
                mask_labels = []
                
                for proposals_per_image, matched_idxs_per_image, labels_per_image in zip(proposals, matched_idxs, labels):
                    pos = torch.where(labels_per_image > 0)[0]
                    mask_proposals.append(proposals_per_image[pos])
                    pos_matched_idxs.append(matched_idxs_per_image[pos])
                    mask_labels.append(labels_per_image[pos])
                
                if len(mask_proposals) == 0:
                    losses.update(dict(roi_mask_loss=torch.tensor(0)))
                    return result, losses
            else:
                mask_proposals = result['boxes']
                
                if len(mask_proposals) == 0:
                    result.update(dict(masks=[torch.empty((0, 28, 28)) for _ in range(len(mask_proposals))]))
                    return result, losses
                
            mask_features = self.mask_roi_pool(features, mask_proposals, image_shapes)
            mask_logits = self.mask_predictor(mask_features)
            
            if self.training:
                gt_masks = [target['masks'] for target in targets]
                mask_loss = maskrcnn_loss(mask_logits, mask_proposals, pos_matched_idxs, mask_labels, gt_masks)
                losses.update(dict(roi_mask_loss=mask_loss))
            else:
                labels = result['labels']
                masks_probs = []
                for mask_logit, label in zip(mask_logits, labels):
                    mask_prob = mask_logit[torch.arange(label.shape[0]), label].sigmoid()
                    masks_probs.append(mask_prob)
                result.update(dict(masks=masks_probs))
                
        return result, losses 

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss