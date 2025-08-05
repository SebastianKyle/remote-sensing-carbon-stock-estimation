import math

import torch
import torch.nn.functional as F


class Transformer:
    def __init__(self, min_size, max_size, image_mean, image_std, resize_to=None):
        self.min_size = min_size
        self.max_size = max_size
        self.image_mean = image_mean
        self.image_std = image_std
        self.resize_to = resize_to
        
    def __call__(self, images, targets=None):
        if not isinstance(images, (list, tuple)):
            images = [images]
            targets = [targets] if targets is not None else None

        # Process each image and target
        processed_images = []
        processed_targets = []
        image_sizes = []

        for i, image in enumerate(images):
            image = self.normalize(image)
            target = targets[i] if targets is not None else None
            image, target = self.resize(image, target)
            processed_images.append(image)
            if target is not None:
                processed_targets.append(target)
            image_sizes.append(image.shape[-2:])

        # Batch the images
        max_size = tuple(max(s) for s in zip(*[img.shape[-2:] for img in processed_images]))
        batch_shape = (len(processed_images), processed_images[0].shape[0]) + max_size
        batched_imgs = torch.zeros(batch_shape, dtype=processed_images[0].dtype, device=processed_images[0].device)
        
        for img, pad_img in zip(processed_images, batched_imgs):
            pad_img[:, :img.shape[-2], :img.shape[-1]].copy_(img)

        if targets is not None:
            return batched_imgs, processed_targets
        return batched_imgs, None

    def normalize(self, image):
        # Scale to [0, 1] if not already
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)

        # Normalize to [0, 1] if not already
        if image.max() > 1.0:
            image = image / 255.0
        
        dtype, device = image.dtype, image.device
        mean = torch.tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.tensor(self.image_std, dtype=dtype, device=device)
        return (image - mean[:, None, None]) / std[:, None, None]

    def resize(self, image, target):
        # print(target)
        ori_image_shape = image.shape[-2:]
        if self.resize_to:
            size = self.resize_to
        else:
            min_size = float(min(image.shape[-2:]))
            max_size = float(max(image.shape[-2:]))
            
            scale_factor = min(self.min_size / min_size, self.max_size / max_size)
            size = [round(s * scale_factor) for s in ori_image_shape]

        image = F.interpolate(image[None], size=size, mode='bilinear', align_corners=False)[0]

        if target is None:
            return image, target
        
        box = target['boxes']
        if box.shape[0] > 0:
            box[:, [0, 2]] = box[:, [0, 2]] * image.shape[-1] / ori_image_shape[1]
            box[:, [1, 3]] = box[:, [1, 3]] * image.shape[-2] / ori_image_shape[0]
            target['boxes'] = box
        
        if 'masks' in target:
            mask = target['masks']
            mask = F.interpolate(mask[None].float(), size=size)[0].byte()
            target['masks'] = mask
            
        return image, target
    
    def postprocess(self, result, image_shapes, ori_image_shapes):
        if isinstance(result, dict) and 'boxes' in result:
            boxes = result['boxes']
            if isinstance(boxes, list):  # Handle batch of results
                processed_results = []
                for box, image_shape, ori_shape in zip(boxes, image_shapes, ori_image_shapes):
                    res = {'boxes': box}
                    if 'labels' in result:
                        res['labels'] = result['labels'][len(processed_results)]
                    if 'scores' in result:
                        res['scores'] = result['scores'][len(processed_results)]
                    if 'masks' in result:
                        res['masks'] = result['masks'][len(processed_results)]

                    res = self._postprocess_single(res, image_shape, ori_shape)
                    processed_results.append(res)
                return processed_results
            else:  # Handle single result
                return self._postprocess_single(result, image_shapes[0], ori_image_shapes[0])
        return result

    def _postprocess_single(self, result, image_shape, ori_image_shape):
        box = result['boxes']
        box[:, [0, 2]] = box[:, [0, 2]] * ori_image_shape[1] / image_shape[1]
        box[:, [1, 3]] = box[:, [1, 3]] * ori_image_shape[0] / image_shape[0]
        result['boxes'] = box
        
        if 'masks' in result:
            mask = result['masks']
            mask = paste_masks_in_image(mask, box, 1, ori_image_shape)
            result['masks'] = mask
            
        return result


def expand_detection(mask, box, padding):
    M = mask.shape[-1]
    scale = (M + 2 * padding) / M
    padded_mask = torch.nn.functional.pad(mask, (padding,) * 4)
    
    w_half = (box[:, 2] - box[:, 0]) * 0.5
    h_half = (box[:, 3] - box[:, 1]) * 0.5
    x_c = (box[:, 2] + box[:, 0]) * 0.5
    y_c = (box[:, 3] + box[:, 1]) * 0.5

    w_half = w_half * scale
    h_half = h_half * scale

    box_exp = torch.zeros_like(box)
    box_exp[:, 0] = x_c - w_half
    box_exp[:, 2] = x_c + w_half
    box_exp[:, 1] = y_c - h_half
    box_exp[:, 3] = y_c + h_half
    return padded_mask, box_exp.to(torch.int64)


def paste_masks_in_image(mask, box, padding, image_shape):
    mask, box = expand_detection(mask, box, padding)
    
    N = mask.shape[0]
    size = (N,) + tuple(image_shape)
    im_mask = torch.zeros(size, dtype=mask.dtype, device=mask.device)
    for m, b, im in zip(mask, box, im_mask):
        b = b.tolist()
        w = max(b[2] - b[0], 1)
        h = max(b[3] - b[1], 1)
        
        m = F.interpolate(m[None, None], size=(h, w), mode='bilinear', align_corners=False)[0][0]

        x1 = max(b[0], 0)
        y1 = max(b[1], 0)
        x2 = min(b[2], image_shape[1])
        y2 = min(b[3], image_shape[0])

        im[y1:y2, x1:x2] = m[(y1 - b[1]):(y2 - b[1]), (x1 - b[0]):(x2 - b[0])]
    return im_mask