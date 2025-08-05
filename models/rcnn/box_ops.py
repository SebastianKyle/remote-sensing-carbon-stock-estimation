import math
from typing import Tuple

import torch
from torch import Tensor
import torchvision

class BoxCoder:
    def __init__(self, weights, bbox_xform_clip=math.log(1000. / 16)):
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    def encode(self, reference_box, proposal):
        """
        Encode a set of proposals with respect to some
        reference boxes

        Arguments:
            reference_boxes (Tensor[N, 4]): reference boxes
            proposals (Tensor[N, 4]): boxes to be encoded
        """
        
        width = proposal[:, 2] - proposal[:, 0]
        height = proposal[:, 3] - proposal[:, 1]
        ctr_x = proposal[:, 0] + 0.5 * width
        ctr_y = proposal[:, 1] + 0.5 * height

        gt_width = reference_box[:, 2] - reference_box[:, 0]
        gt_height = reference_box[:, 3] - reference_box[:, 1]
        gt_ctr_x = reference_box[:, 0] + 0.5 * gt_width
        gt_ctr_y = reference_box[:, 1] + 0.5 * gt_height

        dx = self.weights[0] * (gt_ctr_x - ctr_x) / width
        dy = self.weights[1] * (gt_ctr_y - ctr_y) / height
        dw = self.weights[2] * torch.log(gt_width / width)
        dh = self.weights[3] * torch.log(gt_height / height)

        delta = torch.stack((dx, dy, dw, dh), dim=1)
        return delta

    def decode(self, delta, box):
        """
        From a set of original boxes and encoded relative box offsets,
        get the decoded boxes.

        Arguments:
            delta (Tensor[N, 4]): encoded boxes.
            boxes (Tensor[N, 4]): reference boxes.
        """
        
        dx = delta[:, 0] / self.weights[0]
        dy = delta[:, 1] / self.weights[1]
        dw = delta[:, 2] / self.weights[2]
        dh = delta[:, 3] / self.weights[3]

        dw = torch.clamp(dw, max=self.bbox_xform_clip)
        dh = torch.clamp(dh, max=self.bbox_xform_clip)

        width = box[:, 2] - box[:, 0]
        height = box[:, 3] - box[:, 1]
        ctr_x = box[:, 0] + 0.5 * width
        ctr_y = box[:, 1] + 0.5 * height

        pred_ctr_x = dx * width + ctr_x
        pred_ctr_y = dy * height + ctr_y
        pred_w = torch.exp(dw) * width
        pred_h = torch.exp(dh) * height

        xmin = pred_ctr_x - 0.5 * pred_w
        ymin = pred_ctr_y - 0.5 * pred_h
        xmax = pred_ctr_x + 0.5 * pred_w
        ymax = pred_ctr_y + 0.5 * pred_h

        target = torch.stack((xmin, ymin, xmax, ymax), dim=1)
        return target

def batched_nms(
    boxes: Tensor,
    scores: Tensor,
    idxs: Tensor,
    iou_threshold: float,
) -> Tensor:
    """
    Performs non-maximum suppression in a batched fashion.

    Each index value correspond to a category, and NMS
    will not be applied between elements of different categories.

    Args:
        boxes (Tensor[N, 4]): boxes where NMS will be performed. They
            are expected to be in ``(x1, y1, x2, y2)`` format with ``0 <= x1 < x2`` and
            ``0 <= y1 < y2``.
        scores (Tensor[N]): scores for each one of the boxes
        idxs (Tensor[N]): indices of the categories for each one of the boxes.
        iou_threshold (float): discards all overlapping boxes with IoU > iou_threshold

    Returns:
        Tensor: int64 tensor with the indices of the elements that have been kept by NMS, sorted
        in decreasing order of scores
    """
    if boxes.numel() > (4000 if boxes.device.type == "cpu" else 20000) and not torchvision._is_tracing():
        return _batched_nms_vanilla(boxes, scores, idxs, iou_threshold)
    else:
        return _batched_nms_coordinate_trick(boxes, scores, idxs, iou_threshold)

def _batched_nms_coordinate_trick(
    boxes: Tensor,
    scores: Tensor,
    idxs: Tensor,
    iou_threshold: float,
) -> Tensor:
    # strategy: in order to perform NMS independently per class,
    # we add an offset to all the boxes. The offset is dependent
    # only on the class idx, and is large enough so that boxes
    # from different classes do not overlap
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    max_coordinate = boxes.max()
    offsets = idxs.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
    boxes_for_nms = boxes + offsets[:, None]
    keep = nms(boxes_for_nms, scores, iou_threshold)
    return keep


def _batched_nms_vanilla(
    boxes: Tensor,
    scores: Tensor,
    idxs: Tensor,
    iou_threshold: float,
) -> Tensor:
    # Based on Detectron2 implementation, just manually call nms() on each class independently
    keep_mask = torch.zeros_like(scores, dtype=torch.bool)
    for class_id in torch.unique(idxs):
        curr_indices = torch.where(idxs == class_id)[0]
        curr_keep_indices = nms(boxes[curr_indices], scores[curr_indices], iou_threshold)
        keep_mask[curr_indices[curr_keep_indices]] = True
    keep_indices = torch.where(keep_mask)[0]
    return keep_indices[scores[keep_indices].sort(descending=True)[1]]

def clip_boxes_to_image(boxes: Tensor, size: Tuple[int, int]) -> Tensor:
    """
    Clip boxes so that they lie inside an image of size ``size``.

    .. note::
        For clipping a :class:`~torchvision.tv_tensors.BoundingBoxes` object, consider using
        the transform :func:`~torchvision.transforms.v2.ClampBoundingBoxes` instead.

    Args:
        boxes (Tensor[N, 4]): boxes in ``(x1, y1, x2, y2)`` format
            with ``0 <= x1 < x2`` and ``0 <= y1 < y2``.
        size (Tuple[height, width]): size of the image

    Returns:
        Tensor[N, 4]: clipped boxes
    """
    dim = boxes.dim()
    boxes_x = boxes[..., 0::2]
    boxes_y = boxes[..., 1::2]
    height, width = size

    if torchvision._is_tracing():
        boxes_x = torch.max(boxes_x, torch.tensor(0, dtype=boxes.dtype, device=boxes.device))
        boxes_x = torch.min(boxes_x, torch.tensor(width, dtype=boxes.dtype, device=boxes.device))
        boxes_y = torch.max(boxes_y, torch.tensor(0, dtype=boxes.dtype, device=boxes.device))
        boxes_y = torch.min(boxes_y, torch.tensor(height, dtype=boxes.dtype, device=boxes.device))
    else:
        boxes_x = boxes_x.clamp(min=0, max=width)
        boxes_y = boxes_y.clamp(min=0, max=height)

    clipped_boxes = torch.stack((boxes_x, boxes_y), dim=dim)
    return clipped_boxes.reshape(boxes.shape)
    
def box_iou(box_a, box_b):
    """
    Arguments:
        boxe_a (Tensor[N, 4])
        boxe_b (Tensor[M, 4])

    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in box_a and box_b
    """
    
    lt = torch.max(box_a[:, None, :2], box_b[:, :2])
    rb = torch.min(box_a[:, None, 2:], box_b[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area_a = torch.prod(box_a[:, 2:] - box_a[:, :2], 1)
    area_b = torch.prod(box_b[:, 2:] - box_b[:, :2], 1)
    
    return inter / (area_a[:, None] + area_b - inter)

def box_ciou(box_a, box_b, eps=1e-7):
    """
    Arguments:
        box_a: (Tensor[N, 4])
        box_b: (Tensor[M, 4])
        eps: (float)

    Returns:
        ciou (Tensor[N, M]): the NxM matrix containing the pairwise
            CIoU values for every element in box_a and box_b
    """

    # IoU
    iou = box_iou(box_a, box_b)  # [N, M]

    # Centers
    a_cx = (box_a[:, 0] + box_a[:, 2]) / 2
    a_cy = (box_a[:, 1] + box_a[:, 3]) / 2
    b_cx = (box_b[:, 0] + box_b[:, 2]) / 2
    b_cy = (box_b[:, 1] + box_b[:, 3]) / 2

    # Center distance squared
    center_dist = (a_cx[:, None] - b_cx[None, :]) ** 2 + (a_cy[:, None] - b_cy[None, :]) ** 2

    # Enclosing box
    enclose_x1 = torch.min(box_a[:, None, 0], box_b[:, 0])
    enclose_y1 = torch.min(box_a[:, None, 1], box_b[:, 1])
    enclose_x2 = torch.max(box_a[:, None, 2], box_b[:, 2])
    enclose_y2 = torch.max(box_a[:, None, 3], box_b[:, 3])
    enclose_diag = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2 + eps

    # Aspect ratio
    w_a = (box_a[:, 2] - box_a[:, 0]).clamp(min=eps)
    h_a = (box_a[:, 3] - box_a[:, 1]).clamp(min=eps)
    w_b = (box_b[:, 2] - box_b[:, 0]).clamp(min=eps)
    h_b = (box_b[:, 3] - box_b[:, 1]).clamp(min=eps)
    v = (4 / (math.pi ** 2)) * (torch.atan(w_b / h_b)[None, :] - torch.atan(w_a / h_a)[:, None]) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    ciou = iou - (center_dist / enclose_diag) - alpha * v
    return ciou
    
def process_box(box, score, image_shape, min_size):
    """
    Clip boxes in the image size and remove boxes which are too small.
    """
    
    box[:, [0, 2]] = box[:, [0, 2]].clamp(0, image_shape[1]) 
    box[:, [1, 3]] = box[:, [1, 3]].clamp(0, image_shape[0]) 

    w, h = box[:, 2] - box[:, 0], box[:, 3] - box[:, 1]
    keep = torch.where((w >= min_size) & (h >= min_size))[0]
    box, score = box[keep], score[keep]
    return box, score


def nms(box, score, threshold):
    """
    Arguments:
        box (Tensor[N, 4])
        score (Tensor[N]): scores of the boxes.
        threshold (float): iou threshold.

    Returns: 
        keep (Tensor): indices of boxes filtered by NMS.
    """
    
    return torch.ops.torchvision.nms(box, score, threshold)
    

# just for test. It is too slow. Don't use it during train
def slow_nms(box, nms_thresh):
    idx = torch.arange(box.size(0))
    
    keep = []
    while idx.size(0) > 0:
        keep.append(idx[0].item())
        head_box = box[idx[0], None, :]
        remain = torch.where(box_iou(head_box, box[idx]) <= nms_thresh)[1]
        idx = idx[remain]
    
    return keep