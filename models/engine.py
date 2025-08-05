import sys
import time

import torch
torch.autograd.set_detect_anomaly(True)
import numpy as np
import torchvision

from .utils import Meter, TextArea
from .detr.box_ops import box_xyxy_to_cxcywh
from .detr import DETRLoss
from .rcnn.box_ops import box_ciou
from utils.misc import nested_tensor_from_tensor_list
try:
    from .datasets import CocoEvaluator, prepare_for_coco
except:
    pass


def train_one_epoch(model, optimizer, data_loader, device, epoch, args, detr_criterion=None, train_chm=False):
    # Get current learning rate from optimizer
    current_lr = optimizer.param_groups[0]['lr']
    for p in optimizer.param_groups:
        p["lr"] = current_lr

    iters = len(data_loader) if args.iters < 0 else args.iters

    t_m = Meter("total")
    m_m = Meter("model")
    b_m = Meter("backward")
    model.train()
    A = time.time()

    total_loss_value = 0

    if detr_criterion is not None:
        detr_criterion.train()

    # How to handle the edge image in the training loop 
    for i, (rgb_imgs, chm_imgs, targets) in enumerate(data_loader):
        T = time.time()
        num_iters = epoch * len(data_loader) + i
        
        # Move all images and targets to device
        rgb_imgs = [img.to(device) for img in rgb_imgs]
        chm_imgs = [img.to(device) for img in chm_imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Filter out targets with no boxes
        valid_indices = [i for i, target in enumerate(targets) if target['boxes'].shape[0] > 0]
        if not valid_indices:
            continue

        rgb_imgs = [rgb_imgs[i] for i in valid_indices]
        chm_imgs = [chm_imgs[i] for i in valid_indices]
        targets = [targets[i] for i in valid_indices]

        batch_size = len(rgb_imgs)
                
        S = time.time()
        
        if args.model in ['faster_rcnn', 'swin_rcnn', 'retinanet']:
            if args.frcnn_input == 'rgb':
                losses = model(rgb_imgs, targets)
            elif args.frcnn_input == 'chm':
                losses = model(chm_imgs, targets)
            elif args.frcnn_input == 'both':
                inputs = []
                for rgb_img, chm_img in zip(rgb_imgs, chm_imgs):
                    # Ensure chm_img has shape (1, H, W)
                    if chm_img.dim() == 2:
                        chm_img = chm_img.unsqueeze(0)
                    # Concatenate along channel dimension: (3, H, W) + (1, H, W) -> (4, H, W)
                    # input_4ch = torch.cat([rgb_img, chm_img], dim=0)
                    # inputs.append(input_4ch)
                    input = rgb_img + 0.1 * chm_img.repeat(3, 1, 1)
                    inputs.append(input)
                losses = model(inputs, targets)
            total_loss = sum(losses.values())
        elif args.model in ['ca_rcnn', 'cba_rcnn', 'lka_rcnn', 'msca_rcnn', 'aattn_rcnn', 'cba_convnext_rcnn', 'hcf_rcnn', 'hcf_rcnn_no_fpn', 'hcf_rcnn_efficientnet', 'hcf_rcnn_swin_efficientnet', 'hcf_lka_rcnn', 'hcf_ca_rcnn', 'hcf_rf_rcnn', 'cor_rcnn', 'caf_rcnn', 'transcba_rcnn']:
            losses = model(rgb_imgs, chm_imgs, targets)
            lw = args.loss_weights
            total_loss = (
                lw.get('rpn_objectness', 1.0) * losses['rpn_objectness_loss'] +
                lw.get('rpn_box', 1.0) * losses['rpn_box_loss'] +
                lw.get('roi_classifier', 0.8) * losses['roi_classifier_loss'] +
                lw.get('roi_box', 1.2) * losses['roi_box_loss']
            )
        elif args.model in ['fusion_retinanet']:
            losses = model(rgb_imgs, chm_imgs, targets)
            total_loss = losses['classification'] + losses['bbox_regression']
        elif args.model == 'detr':
            # Forward pass
            if train_chm:
                outputs = model(chm_imgs)
            else:
                outputs = model(rgb_imgs)
            
            # Convert target boxes to normalized (cx,cy,w,h) format for loss computation
            processed_targets = []
            for target, img in zip(targets, rgb_imgs):
                # Get image size for normalization
                h, w = img.shape[-2:]
                
                # Normalize boxes to [0, 1] range
                boxes = target['boxes'].float()  # [num_boxes, 4] in (x1,y1,x2,y2) format
                boxes[:, [0, 2]] /= w  # normalize x coordinates
                boxes[:, [1, 3]] /= h  # normalize y coordinates
                
                # Convert normalized boxes to (cx,cy,w,h) format
                boxes_cxcywh = box_xyxy_to_cxcywh(boxes)
                
                processed_targets.append({
                    'labels': target['labels'],
                    'boxes': boxes_cxcywh
                })
            
            # Calculate loss - model outputs should already be in correct format
            loss_dict = detr_criterion(outputs, processed_targets)
            weight_dict = detr_criterion.weight_dict
            total_loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            losses = {k: v.item() for k, v in loss_dict.items()}
        elif args.model == 'cba_detr':
            # Forward pass
            outputs = model(rgb_imgs, chm_imgs)
            
            # Convert target boxes to normalized (cx,cy,w,h) format for loss computation
            processed_targets = []
            for target, img in zip(targets, rgb_imgs):
                # Get image size for normalization
                h, w = img.shape[-2:]
                
                # Normalize boxes to [0, 1] range
                boxes = target['boxes'].float()  # [num_boxes, 4] in (x1,y1,x2,y2) format
                boxes[:, [0, 2]] /= w  # normalize x coordinates
                boxes[:, [1, 3]] /= h  # normalize y coordinates
                
                # Convert normalized boxes to (cx,cy,w,h) format
                boxes_cxcywh = box_xyxy_to_cxcywh(boxes)
                
                processed_targets.append({
                    'labels': target['labels'],
                    'boxes': boxes_cxcywh
                })
            
            # Calculate loss - model outputs should already be in correct format
            loss_dict = detr_criterion(outputs, processed_targets)
            weight_dict = detr_criterion.weight_dict
            total_loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            losses = {k: v.item() for k, v in loss_dict.items()} 
        elif args.model == 'yolov12':
            if train_chm:
                losses = model(chm_imgs, targets)
            else:
                losses = model(rgb_imgs, targets)
            total_loss = sum(losses.values()) * batch_size
        elif args.model == 'fusion_yolov12':
            losses = model(rgb_imgs, chm_imgs, targets)
            total_loss = sum(losses.values()) * batch_size
            
        total_loss_value += total_loss.item()
        m_m.update(time.time() - S)
            
        S = time.time()
        total_loss.backward()
        b_m.update(time.time() - S)
        
        # Gradient clipping for DETR
        if args.model == 'detr' and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        
        optimizer.step()
        optimizer.zero_grad()

        if num_iters % args.print_freq == 0:
            print("{}\t".format(num_iters), "\t".join("{:.3f}".format(l if isinstance(l, float) else l.item()) for l in losses.values()))

        t_m.update(time.time() - T)
        if i >= iters - 1:
            break
           
    A = time.time() - A
    print("iter: {:.1f}, total: {:.1f}, model: {:.1f}, backward: {:.1f}".format(1000*A/iters,1000*t_m.avg,1000*m_m.avg,1000*b_m.avg))
    return A / iters, total_loss_value / iters
            

def evaluate(model, data_loader, device, args, generate=True):
    iter_eval = None
    if generate:
        iter_eval = generate_results(model, data_loader, device, args)

    dataset = data_loader #
    iou_types = ["bbox", "segm"]
    coco_evaluator = CocoEvaluator(dataset.coco, iou_types)

    results = torch.load(args.results, map_location="cpu")

    S = time.time()
    coco_evaluator.accumulate(results)
    print("accumulate: {:.1f}s".format(time.time() - S))

    # collect outputs of buildin function print
    temp = sys.stdout
    sys.stdout = TextArea()

    coco_evaluator.summarize()

    output = sys.stdout
    sys.stdout = temp
        
    return output, iter_eval

def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)

    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area
    return iou

def calculate_ciou(box1, box2, eps=1e-7):
    # box1, box2: [x1, y1, x2, y2]
    # Returns scalar ciou
    import math
    # IoU
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)
    union_area = box1_area + box2_area - inter_area + eps
    iou = inter_area / union_area

    # Center distance
    px = (x1 + x2) / 2
    py = (y1 + y2) / 2
    gx = (x1g + x2g) / 2
    gy = (y1g + y2g) / 2
    center_dist = (px - gx) ** 2 + (py - gy) ** 2

    # Enclosing box diagonal
    enclose_x1 = min(x1, x1g)
    enclose_y1 = min(y1, y1g)
    enclose_x2 = max(x2, x2g)
    enclose_y2 = max(y2, y2g)
    diag = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2 + eps

    # Aspect ratio
    w_pred = max(x2 - x1, eps)
    h_pred = max(y2 - y1, eps)
    w_gt = max(x2g - x1g, eps)
    h_gt = max(y2g - y1g, eps)
    v = (4 / (math.pi ** 2)) * (math.atan(w_gt / h_gt) - math.atan(w_pred / h_pred)) ** 2
    alpha = v / (1 - iou + v + eps)
    ciou = iou - (center_dist / diag) - alpha * v
    return ciou

def evaluate_model(model, data_loader, device, iou_threshold=0.5, generate=True, args=None, use_chm=False):
    # Ensure model is in evaluation mode
    model.eval()
    
    all_tp = 0
    all_fp = 0
    all_fn = 0
    all_iou = []

    all_true_boxes_per_image = []
    all_pred_boxes_per_image = []
    all_scores_per_image = []

    all_diou = []
    all_ciou = []

    with torch.no_grad():
        for rgb_imgs, chm_imgs, targets in data_loader:
            # Move batch to device
            rgb_imgs = [img.to(device) for img in rgb_imgs]
            chm_imgs = [img.to(device) for img in chm_imgs]
            
            # Get model output based on model type
            if args.model in ['faster_rcnn', 'swin_rcnn', 'yolov12', 'retinanet']:
                if args.frcnn_input == 'rgb':
                    outputs = model(rgb_imgs)
                elif args.frcnn_input == 'chm':
                    outputs = model(chm_imgs)
                elif args.frcnn_input == 'both':                
                    inputs = []
                    for rgb_img, chm_img in zip(rgb_imgs, chm_imgs):
                        # Ensure chm_img has shape (1, H, W)
                        if chm_img.dim() == 2:
                            chm_img = chm_img.unsqueeze(0)
                        # Concatenate along channel dimension: (3, H, W) + (1, H, W) -> (4, H, W)
                        # input_4ch = torch.cat([rgb_img, chm_img], dim=0)
                        # inputs.append(input_4ch)
                        input = rgb_img + 0.1 * chm_img.repeat(3, 1, 1)
                        inputs.append(input)
                    
                    outputs = model(inputs)
            elif args.model in ['ca_rcnn', 'cba_rcnn', 'lka_rcnn', 'msca_rcnn', 'attn_rcnn', 'cba_detr', 'cba_convnext_rcnn', 'fusion_yolov12', 'fusion_retinanet', 'hcf_rcnn', 'hcf_rcnn_no_fpn', 'hcf_rcnn_efficientnet', 'hcf_rcnn_swin_efficientnet', 'hcf_lka_rcnn', 'hcf_ca_rcnn', 'hcf_rf_rcnn', 'cor_rcnn', 'caf_rcnn', 'transcba_rcnn']:
                outputs = model(rgb_imgs, chm_imgs)
            elif args.model == 'detr':
                if use_chm:
                    outputs = model(chm_imgs)
                else:
                    outputs = model(rgb_imgs)
            elif args.model.lower().startswith('deepforest'):
                # DeepForest models (both RetinaNet and Faster R-CNN)
                outputs = model(rgb_imgs)
                # print(f"outputs: {outputs}")
            
            # Process each image in the batch
            for idx, target in enumerate(targets):
                true_boxes = target["boxes"].cpu().numpy()
                output = outputs[idx] if isinstance(outputs, list) else outputs
                
                # Handle different model outputs
                if args.model.lower().startswith('deepforest'):
                    pred_boxes = output["boxes"].cpu().numpy()
                    scores = output["scores"].cpu().numpy()
                elif args.model in ['detr', 'cba_detr']:
                    # Get predictions and scores
                    pred_boxes = output["boxes"].cpu().numpy()
                    scores = output["scores"][:, 0].cpu().numpy()
                    
                    # Filter predictions based on confidence threshold
                    keep = scores > 0.1  # Confidence threshold
                    pred_boxes = pred_boxes[keep]
                    scores = scores[keep]
                    
                    # Sort by confidence
                    sorted_idx = np.argsort(scores)[::-1]
                    pred_boxes = pred_boxes[sorted_idx]
                    scores = scores[sorted_idx]
                    
                    # Apply NMS
                    if len(pred_boxes) > 0:
                        # Convert to tensor for NMS
                        boxes_tensor = torch.from_numpy(pred_boxes).float()
                        scores_tensor = torch.from_numpy(scores).float()
                        
                        # Apply NMS
                        keep_indices = torchvision.ops.nms(boxes_tensor, scores_tensor, 0.2)
                        pred_boxes = pred_boxes[keep_indices.numpy()]
                        scores = scores[keep_indices.numpy()]    
                elif args.model in ['yolov12', 'fusion_yolov12']:
                    pred_boxes = output["boxes"].cpu().numpy()
                    scores = output["scores"][:, 0].cpu().numpy()

                    keep = scores > 0.1
                    pred_boxes = pred_boxes[keep]
                    scores = scores[keep]
                    
                    # Sort by confidence
                    sorted_idx = np.argsort(scores)[::-1]
                    pred_boxes = pred_boxes[sorted_idx]
                    scores = scores[sorted_idx]
                    
                    # Apply NMS
                    if len(pred_boxes) > 0:
                        # Convert to tensor for NMS
                        boxes_tensor = torch.from_numpy(pred_boxes).float()
                        scores_tensor = torch.from_numpy(scores).float()
                        
                        # Apply NMS
                        keep_indices = torchvision.ops.nms(boxes_tensor, scores_tensor, 0.2)
                        pred_boxes = pred_boxes[keep_indices.numpy()]
                        scores = scores[keep_indices.numpy()]
                else:
                    # Other models
                    pred_boxes = output["boxes"].cpu().numpy()
                    scores = output["scores"].cpu().numpy()

                all_true_boxes_per_image.append(true_boxes)
                all_pred_boxes_per_image.append(pred_boxes)
                all_scores_per_image.append(scores)

                # Skip if no predictions or ground truth
                if len(pred_boxes) == 0 or len(true_boxes) == 0:
                    continue

                # Unique matching for this image using a greedy approach
                sorted_indices = np.argsort(scores)[::-1]  # Sort by descending confidence
                pred_boxes_sorted = pred_boxes[sorted_indices]
                pred_scores_sorted = scores[sorted_indices]

                pred_boxes_tensor = torch.from_numpy(pred_boxes_sorted).float()
                pred_scores_tensor = torch.from_numpy(pred_scores_sorted).float()
                true_boxes_tensor = torch.from_numpy(true_boxes).float()
                iou_matrix = torchvision.ops.box_iou(pred_boxes_tensor, true_boxes_tensor)

                matched, mean_iou = match_predictions(
                    pred_classes=torch.zeros(len(pred_boxes_sorted), dtype=torch.long), 
                    true_classes=torch.zeros(len(true_boxes), dtype=torch.long), 
                    iou=iou_matrix
                )

                # ciou_matrix = box_ciou(pred_boxes_tensor, true_boxes_tensor)
                # matched, mean_iou, mean_ciou = match_predictions_hybrid(
                #     pred_classes=torch.zeros(len(pred_boxes_sorted), dtype=torch.long),
                #     true_classes=torch.zeros(len(true_boxes), dtype=torch.long),
                #     iou=iou_matrix,
                #     ciou=ciou_matrix,
                #     iou_thresh=0.5,
                #     ciou_thresh=0.5
                # )

                all_tp += matched.sum().item()
                all_fp += len(pred_boxes) - matched.sum().item()
                all_fn += len(true_boxes) - matched.sum().item()
                all_iou.append(mean_iou)
                # all_ciou.append(mean_ciou)

                # Compute DIoU/CIoU for matched pairs
                # if matched.sum().item() > 0:
                #     matched_pred_boxes = pred_boxes_tensor[matched]
                #     matched_true_boxes = true_boxes_tensor[:matched.sum().item()]
                #     all_diou.append(diou_metric(matched_pred_boxes, matched_true_boxes))
                #     all_ciou.append(ciou_metric(matched_pred_boxes, matched_true_boxes))
                # else:
                #     all_diou.append(0.0)
                #     all_ciou.append(0.0)
                
    precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
    recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    mean_iou = np.mean(all_iou) if len(all_iou) > 0 else 0
    # mean_diou = np.mean(all_diou) if len(all_diou) > 0 else 0
    # mean_ciou = np.mean(all_ciou) if len(all_ciou) > 0 else 0
    
    # Compute AP per image and take mean AP (mAP)
    # average_precision = compute_average_precision(
    #     all_true_boxes_per_image, 
    #     all_pred_boxes_per_image, 
    #     all_scores_per_image, 
    #     iou_threshold
    # )

    average_precision = compute_average_precision(
        all_true_boxes_per_image,
        all_pred_boxes_per_image,
        all_scores_per_image,
        iou_threshold=iou_threshold,
        # ciou_threshold=0.5
    )
    
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "mean_iou": mean_iou,
        # "mean_diou": mean_diou,
        # "mean_ciou": mean_ciou,
        "average_precision": average_precision
    }

def match_predictions(pred_classes, true_classes, iou):
    """
    Matches predictions to ground truth using IoU.

    Args:
        pred_classes (torch.Tensor): Shape (N,). Predicted class indices.
        true_classes (torch.Tensor): Shape (M,). Target class indices.
        iou (torch.Tensor): Shape (N, M). IoU values for each (pred, target) pair.

    Returns:
        torch.Tensor: Boolean tensor (N,) where each prediction is marked as True if matched.
    """
    correct = np.zeros(pred_classes.shape[0], dtype=bool)  # One match per prediction
    matched_ious = []

    # Step 1: Ensure correct classes
    correct_class = true_classes[:, None] == pred_classes  # (M, N)
    iou = iou * correct_class.T  # Zero out incorrect class matches

    iou = iou.cpu().numpy()
    threshold = 0.5  # Single IoU threshold

    # Step 2: Find valid matches (IoU >= 0.5)
    matches = np.nonzero(iou >= threshold)  # (M, N) --> (matched pairs)
    matches = np.array(matches).T  # Convert to list of matched pairs

    if matches.shape[0]:  # If we have matches
        if matches.shape[0] > 1:
            # Step 3: Sort matches by highest IoU, prioritize better matches
            matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
            # Step 4: Keep only **one match per ground truth**
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            # Step 5: Keep only **one match per prediction**
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]

        correct[matches[:, 0].astype(int)] = True  # Mark the predictions as correct
        matched_ious = iou[matches[:, 0], matches[:, 1]]

    mean_iou = np.mean(matched_ious) if len(matched_ious) > 0 else 0.0

    return torch.tensor(correct, dtype=torch.bool, device=pred_classes.device), mean_iou

def match_predictions_hybrid(pred_classes, true_classes, iou, ciou, iou_thresh=0.5, ciou_thresh=0.5):
    """
    Matches predictions to ground truth using both IoU and CIoU thresholds.

    Args:
        pred_classes (torch.Tensor): Shape (N,). Predicted class indices.
        true_classes (torch.Tensor): Shape (M,). Target class indices.
        iou (torch.Tensor): Shape (N, M). IoU values for each (pred, target) pair.
        ciou (torch.Tensor): Shape (N, M). CIoU values for each (pred, target) pair.
        iou_thresh (float): IoU threshold for matching.
        ciou_thresh (float): CIoU threshold for matching.

    Returns:
        torch.Tensor: Boolean tensor (N,) where each prediction is marked as True if matched.
        float: Mean IoU of matched pairs.
    """
    correct = np.zeros(pred_classes.shape[0], dtype=bool)
    matched_ious = []
    matched_cious = []

    # Step 1: Ensure correct classes
    correct_class = true_classes[:, None] == pred_classes  # (M, N)
    iou = iou * correct_class.T  # Zero out incorrect class matches
    ciou = ciou * correct_class.T

    iou = iou.cpu().numpy()
    ciou = ciou.cpu().numpy()

    # Step 2: Find valid matches (IoU >= iou_thresh and CIoU >= ciou_thresh)
    matches = np.nonzero((iou >= iou_thresh) & (ciou >= ciou_thresh))  # (M, N) --> (matched pairs)
    matches = np.array(matches).T  # Convert to list of matched pairs

    if matches.shape[0]:  # If we have matches
        if matches.shape[0] > 1:

            combined = iou[matches[:, 0], matches[:, 1]] + ciou[matches[:, 0], matches[:, 1]]
            
            # Step 3: Sort matches by highest IoU, prioritize better matches
            # matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
            matches = matches[combined.argsort()[::-1]]
            # Step 4: Keep only **one match per ground truth**
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            # Step 5: Keep only **one match per prediction**
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]

        correct[matches[:, 0].astype(int)] = True  # Mark the predictions as correct
        matched_ious = iou[matches[:, 0], matches[:, 1]]
        matched_cious = ciou[matches[:, 0], matches[:, 1]]

    mean_iou = np.mean(matched_ious) if len(matched_ious) > 0 else 0.0
    mean_ciou = np.mean(matched_cious) if len(matched_cious) > 0 else 0.0

    return torch.tensor(correct, dtype=torch.bool, device=pred_classes.device), mean_iou, mean_ciou

def compute_average_precision(true_boxes_list, pred_boxes_list, scores_list, iou_threshold):
    """
    Compute mean Average Precision (mAP) across all images by accumulating all detections and
    calculating a single precision-recall curve.

    Parameters:
        - true_boxes_list: List of ground truth bounding boxes per image.
        - pred_boxes_list: List of predicted bounding boxes per image.
        - scores_list: List of prediction scores per image.
        - iou_threshold: IoU threshold to consider a prediction a true positive.

    Returns:
        - Average Precision (AP) computed from a single precision-recall curve.
    """

    # Flatten all boxes and scores across images into a single list
    all_true_boxes = []
    all_pred_boxes = []
    all_scores = []
    image_ids = []

    for img_idx, (true_boxes, pred_boxes, scores) in enumerate(zip(true_boxes_list, pred_boxes_list, scores_list)):
        all_true_boxes.extend([(img_idx, box) for box in true_boxes])  # Keep track of image ID
        all_pred_boxes.extend([(img_idx, box) for box in pred_boxes])
        all_scores.extend(scores)

    if len(all_pred_boxes) == 0:
        return 0.0  # No predictions, AP = 0

    # Sort all predictions across images by confidence score (descending)
    sorted_indices = np.argsort(all_scores)[::-1]
    all_pred_boxes = [all_pred_boxes[i] for i in sorted_indices]

    tp = np.zeros(len(all_pred_boxes))
    fp = np.zeros(len(all_pred_boxes))
    matched_gt = set()

    # Evaluate each prediction globally
    for i, (img_id, pred_box) in enumerate(all_pred_boxes):
        best_iou = 0
        best_match_idx = -1

        # Find the best matching GT box in the same image
        for j, (gt_img_id, true_box) in enumerate(all_true_boxes):
            if gt_img_id != img_id:
                continue  # Only match GT boxes in the same image

            if j in matched_gt:
                continue  # Skip already matched GT boxes

            iou = calculate_iou(pred_box, true_box)
            if iou > best_iou:
                best_iou = iou
                best_match_idx = j

        # Assign TP or FP based on IoU threshold
        if best_iou >= iou_threshold:
            tp[i] = 1
            matched_gt.add(best_match_idx)  # Mark this GT box as matched
        else:
            fp[i] = 1

    # Compute precision-recall curve across all images
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    precision = tp_cumsum / (tp_cumsum + fp_cumsum)
    recall = tp_cumsum / len(all_true_boxes) if len(all_true_boxes) > 0 else np.zeros_like(tp_cumsum)

    # Ensure precision is decreasing for proper AP calculation
    precision = np.concatenate(([0], precision, [0]))
    recall = np.concatenate(([0], recall, [1]))
    precision = np.maximum.accumulate(precision[::-1])[::-1]

    # Compute AP as area under the single precision-recall curve
    ap = np.trapz(precision, recall)
    return ap

def compute_average_precision_hybrid(true_boxes_list, pred_boxes_list, scores_list, iou_threshold, ciou_threshold):
    all_true_boxes = []
    all_pred_boxes = []
    all_scores = []
    image_ids = []

    for img_idx, (true_boxes, pred_boxes, scores) in enumerate(zip(true_boxes_list, pred_boxes_list, scores_list)):
        all_true_boxes.extend([(img_idx, box) for box in true_boxes])
        all_pred_boxes.extend([(img_idx, box) for box in pred_boxes])
        all_scores.extend(scores)

    if len(all_pred_boxes) == 0:
        return 0.0

    sorted_indices = np.argsort(all_scores)[::-1]
    all_pred_boxes = [all_pred_boxes[i] for i in sorted_indices]

    tp = np.zeros(len(all_pred_boxes))
    fp = np.zeros(len(all_pred_boxes))
    matched_gt = set()

    for i, (img_id, pred_box) in enumerate(all_pred_boxes):
        best_iou = 0
        best_ciou = 0
        best_match_idx = -1

        for j, (gt_img_id, true_box) in enumerate(all_true_boxes):
            if gt_img_id != img_id or j in matched_gt:
                continue

            iou = calculate_iou(pred_box, true_box)
            ciou = calculate_ciou(pred_box, true_box)

            if iou >= iou_threshold and ciou >= ciou_threshold and iou > best_iou:
                best_iou = iou
                best_ciou = ciou
                best_match_idx = j

        if best_match_idx >= 0:
            tp[i] = 1
            matched_gt.add(best_match_idx)
        else:
            fp[i] = 1

    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    precision = tp_cumsum / (tp_cumsum + fp_cumsum)
    recall = tp_cumsum / len(all_true_boxes) if len(all_true_boxes) > 0 else np.zeros_like(tp_cumsum)

    precision = np.concatenate(([0], precision, [0]))
    recall = np.concatenate(([0], recall, [1]))
    precision = np.maximum.accumulate(precision[::-1])[::-1]

    ap = np.trapz(precision, recall)
    return ap
 
# generate results file   
@torch.no_grad()   
def generate_results(model, data_loader, device, args):
    iters = len(data_loader) if args.iters < 0 else args.iters
        
    t_m = Meter("total")
    m_m = Meter("model")
    coco_results = []
    model.eval()
    A = time.time()
    for i, (image, target) in enumerate(data_loader):
        T = time.time()
        
        image = image.to(device)
        target = {k: v.to(device) for k, v in target.items()}

        S = time.time()
        #torch.cuda.synchronize()
        output = model(image)
        m_m.update(time.time() - S)
        
        prediction = {target["image_id"].item(): {k: v.cpu() for k, v in output.items()}}
        coco_results.extend(prepare_for_coco(prediction))

        t_m.update(time.time() - T)
        if i >= iters - 1:
            break
     
    A = time.time() - A 
    print("iter: {:.1f}, total: {:.1f}, model: {:.1f}".format(1000*A/iters,1000*t_m.avg,1000*m_m.avg))
    torch.save(coco_results, args.results)
        
    return A / iters

def diou_metric(pred_boxes, gt_boxes):
    # pred_boxes, gt_boxes: [N, 4] torch.Tensor
    if pred_boxes.numel() == 0 or gt_boxes.numel() == 0:
        return 0.0
    iou = torchvision.ops.box_iou(pred_boxes, gt_boxes).diag()
    px = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2
    py = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2
    gx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2
    gy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2
    center_dist = (px - gx) ** 2 + (py - gy) ** 2
    x1 = torch.min(pred_boxes[:, 0], gt_boxes[:, 0])
    y1 = torch.min(pred_boxes[:, 1], gt_boxes[:, 1])
    x2 = torch.max(pred_boxes[:, 2], gt_boxes[:, 2])
    y2 = torch.max(pred_boxes[:, 3], gt_boxes[:, 3])
    diag = (x2 - x1) ** 2 + (y2 - y1) ** 2 + 1e-7
    diou = iou - center_dist / diag
    return diou.mean().item()

def ciou_metric(pred_boxes, gt_boxes):
    if pred_boxes.numel() == 0 or gt_boxes.numel() == 0:
        return 0.0
    iou = torchvision.ops.box_iou(pred_boxes, gt_boxes).diag()
    px = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2
    py = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2
    gx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2
    gy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2
    center_dist = (px - gx) ** 2 + (py - gy) ** 2
    x1 = torch.min(pred_boxes[:, 0], gt_boxes[:, 0])
    y1 = torch.min(pred_boxes[:, 1], gt_boxes[:, 1])
    x2 = torch.max(pred_boxes[:, 2], gt_boxes[:, 2])
    y2 = torch.max(pred_boxes[:, 3], gt_boxes[:, 3])
    diag = (x2 - x1) ** 2 + (y2 - y1) ** 2 + 1e-7
    w_pred = pred_boxes[:, 2] - pred_boxes[:, 0]
    h_pred = pred_boxes[:, 3] - pred_boxes[:, 1]
    w_gt = gt_boxes[:, 2] - gt_boxes[:, 0]
    h_gt = gt_boxes[:, 3] - gt_boxes[:, 1]
    v = (4 / (3.14159265 ** 2)) * (torch.atan(w_gt / (h_gt + 1e-7)) - torch.atan(w_pred / (h_pred + 1e-7))) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou + v + 1e-7)
    ciou = iou - center_dist / diag - alpha * v
    return ciou.mean().item()