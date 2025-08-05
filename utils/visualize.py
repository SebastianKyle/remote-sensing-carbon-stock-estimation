import matplotlib.pyplot as plt
import matplotlib.patches as patches
# from datasets.dataset import TreeDataset
# from src.datasets.dataset import TreeDataset
import sys
# sys.path.append('../thesis-tree-delineation')
sys.path.append('../src')
from datasets.dataset import TreeDataset
import yaml
import torch
import models
from torchvision import transforms
import os
import torchvision.ops as ops
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage import exposure
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
import rasterio
from skimage import filters, morphology, measure
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
import matplotlib.patches as patches
from models.DeepForest.wrapper import DeepForestWrapper
import cv2

# def visualize_image_with_annotation(dataset, img_id):
#     rgb_image, chm_image = dataset.get_image(img_id)
#     target = dataset.get_target_with_id(img_id)
    
#     fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
#     # Plot RGB image with annotations
#     axes[0].imshow(rgb_image)
#     for bbox in target['boxes']:
#         rect = patches.Rectangle(
#             (bbox[0], bbox[1]),
#             bbox[2] - bbox[0],
#             bbox[3] - bbox[1],
#             linewidth=2,
#             edgecolor='r',
#             facecolor='none'
#         )
#         axes[0].add_patch(rect)
#     axes[0].set_title('RGB Image with Annotations')
#     axes[0].axis('off')
    
#     # Plot CHM image
#     axes[1].imshow(chm_image, cmap='gray')
#     axes[1].set_title('CHM Image')
#     axes[1].axis('off')
    
#     plt.savefig(f"visualizations/{img_id}.png", bbox_inches='tight')
#     plt.show()

def visualize_image_with_annotation(dataset, img_id, mode='rgb', save_path=None):
    """
    Visualize ground truth bounding boxes on either the RGB or CHM image.
    Args:
        dataset: TreeDataset instance
        img_id: Image ID
        mode: 'rgb' or 'chm' (default: 'rgb')
    """
    rgb_image, chm_image = dataset.get_image(img_id)
    target = dataset.get_target_with_id(img_id)
    if mode == 'rgb':
        img = rgb_image
        title = 'Ảnh RGB với hộp đánh nhãn'
        cmap = None
    elif mode == 'chm':
        img = chm_image
        title = 'Ảnh độ cao với hộp đánh nhãn'
        cmap = 'gray'
    else:
        raise ValueError("mode must be 'rgb' or 'chm'")
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(img, cmap=cmap)
    for bbox in target['boxes']:
        rect = patches.Rectangle(
            (bbox[0], bbox[1]),
            bbox[2] - bbox[0],
            bbox[3] - bbox[1],
            linewidth=2,
            edgecolor='yellow',
            facecolor='none'
        )
        ax.add_patch(rect)
    ax.set_title(title)
    ax.axis('off')
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(os.path.join(save_path, f"{img_id}_{mode}"))

    plt.show()

def visualize_image_with_prediction(dataset, img_id, model, device, save_path=None, conf_threshold=0.5, iou_threshold=0.5, config=None):
    rgb_image, chm_image = dataset.get_image(img_id)
    target = dataset.get_target_with_id(img_id)
    rgb_image_tensor = transforms.ToTensor()(rgb_image)
    rgb_img = rgb_image_tensor.to(device)
    chm_img = transforms.ToTensor()(chm_image).to(device)
    model.eval()
    with torch.no_grad():
        if config['test']['model'] in ['faster_rcnn', 'deepforest_retinanet']:
            prediction = model([rgb_img])
        elif config['test']['model'] in ['ca_rcnn', 'cba_rcnn', 'hcf_rcnn']:
            prediction = model([rgb_img], [chm_img])
    boxes = prediction[0]['boxes']
    scores = prediction[0]['scores']
    # Apply confidence threshold
    keep = scores >= conf_threshold
    boxes, scores = boxes[keep], scores[keep]
    # Apply Non-Maximum Suppression (NMS)
    keep_indices = ops.nms(boxes, scores, iou_threshold)
    boxes, scores = boxes[keep_indices], scores[keep_indices]
    # Plot single RGB image with both predictions and ground truth
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(rgb_image)
    # Draw predicted boxes (red)
    for bbox in boxes.cpu().numpy():
        rect = patches.Rectangle(
            (bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
            linewidth=2, edgecolor='r', facecolor='none', label='Prediction'
        )
        ax.add_patch(rect)
    # Draw ground truth boxes (yellow)
    for bbox in target['boxes']:
        rect = patches.Rectangle(
            (bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
            linewidth=2, edgecolor='yellow', facecolor='none', label='Ground Truth'
        )
        ax.add_patch(rect)
    # Create custom legend (avoid duplicate labels)
    handles = [
        patches.Patch(edgecolor='r', facecolor='none', label='Prediction', linewidth=2),
        patches.Patch(edgecolor='yellow', facecolor='none', label='Ground Truth', linewidth=2)
    ]
    # ax.legend(handles=handles, loc='upper right')
    ax.set_title('Dự đoán mô hình HAF R-CNN')
    ax.axis('off')
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.show()

# def visualize_hcf_fusion_attention(dataset, img_id, model, device, 
#                                    rgb_layer_name='backbone.rgb_backbone.layer2', 
#                                    chm_layer_name='backbone.chm_backbone.layer2', 
#                                    hcf_layer_name='backbone.fusion3', 
#                                    alpha=0.3):
#     """
#     Visualize feature map heatmaps (no overlay, no resizing) and RGB with ground truth.
#     """
#     rgb_image, chm_image = dataset.get_image(img_id)
#     target = dataset.get_target_with_id(img_id)

#     # Prepare input tensors
#     rgb_tensor = transforms.ToTensor()(rgb_image).to(device)
#     chm_tensor = transforms.ToTensor()(chm_image).to(device)

#     # Storage for activations
#     activations = {}

#     def get_layer_by_name(model, name):
#         for n, m in model.named_modules():
#             if n == name:
#                 return m
#         raise ValueError(f"Layer {name} not found in model.")

#     # Register hooks
#     hooks = []
#     for key, layer_name in zip(['rgb', 'chm', 'hcf'], [rgb_layer_name, chm_layer_name, hcf_layer_name]):
#         layer = get_layer_by_name(model, layer_name)
#         hooks.append(layer.register_forward_hook(lambda m, i, o, k=key: activations.setdefault(k, o.detach().cpu())))

#     # Forward pass
#     model.eval()
#     with torch.no_grad():
#         _ = model([rgb_tensor], [chm_tensor])

#     # Remove hooks
#     for h in hooks:
#         h.remove()

#     # Helper to process feature map
#     def featuremap_to_heatmap(feat, img_shape):
#         # feat: (B, C, H, W) or (C, H, W)
#         if feat.dim() == 4:
#             feat = feat[0]
#         fmap = feat.mean(0).numpy()  # Average over channels
#         fmap = np.maximum(fmap, 0)
#         fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-8)
#         fmap = cv2.resize(fmap, (img_shape[1], img_shape[0]))

#         heatmap_color = cv2.applyColorMap(np.uint8(255 * fmap), cv2.COLORMAP_JET)

#         return heatmap_color

#     # Generate heatmaps
#     rgb_heatmap = featuremap_to_heatmap(activations['rgb'], rgb_image.shape)
#     chm_heatmap = featuremap_to_heatmap(activations['chm'], rgb_image.shape)
#     hcf_heatmap = featuremap_to_heatmap(activations['hcf'], rgb_image.shape)

#     # Plot: RGB with GT, and 3 heatmaps
#     fig, axes = plt.subplots(1, 4, figsize=(20, 6))

#     # RGB with ground truth
#     axes[0].imshow(rgb_image)
#     for bbox in target['boxes']:
#         rect = patches.Rectangle(
#             (bbox[0], bbox[1]),
#             bbox[2] - bbox[0],
#             bbox[3] - bbox[1],
#             linewidth=2,
#             edgecolor='yellow',
#             facecolor='none'
#         )
#         axes[0].add_patch(rect)
#     axes[0].set_title('RGB + GT')
#     axes[0].axis('off')

#     # Feature map heatmaps
#     for ax, fmap, title in zip(axes[1:], 
#                               [rgb_heatmap, chm_heatmap, hcf_heatmap], 
#                               ['RGB feature', 'CHM feature', 'Fused HCF feature']):
#         ax.imshow(fmap, cmap='jet')
#         ax.set_title(title)
#         ax.axis('off')

#     plt.tight_layout()
#     plt.show()

#     return rgb_heatmap, chm_heatmap, hcf_heatmap

def visualize_hcf_fusion_attention(dataset, img_id, model, device, 
                                   rgb_layer_name='backbone.rgb_backbone.layer2', 
                                   chm_layer_name='backbone.chm_backbone.layer2', 
                                   hcf_layer_name='backbone.fusion3', 
                                   alpha=0.4):
    """
    Visualize attention maps from RGB, CHM, and fused features after first HCF module.
    Args:
        model: HCF R-CNN model
        rgb_image: np.array (H, W, 3)
        chm_image: np.array (H, W)
        device: torch.device
        *_layer_name: str, dot-path to the relevant layer/module
        alpha: float, overlay strength
    """

    rgb_image, chm_image = dataset.get_image(img_id)
    target = dataset.get_target_with_id(img_id)

    # Prepare input tensors
    rgb_tensor = transforms.ToTensor()(rgb_image).to(device)
    chm_tensor = transforms.ToTensor()(chm_image).to(device)

    # Storage for activations
    activations = {}

    def get_layer_by_name(model, name):
        for n, m in model.named_modules():
            if n == name:
                return m
        raise ValueError(f"Layer {name} not found in model.")

    # Register hooks
    hooks = []
    for key, layer_name in zip(['rgb', 'chm', 'hcf'], [rgb_layer_name, chm_layer_name, hcf_layer_name]):
        layer = get_layer_by_name(model, layer_name)
        hooks.append(layer.register_forward_hook(lambda m, i, o, k=key: activations.setdefault(k, o.detach().cpu())))

    # Forward pass
    model.eval()
    with torch.no_grad():
        _ = model([rgb_tensor], [chm_tensor])

    # Remove hooks
    for h in hooks:
        h.remove()

    # Helper to process feature map
    def featuremap_to_heatmap(feat, img_shape):
        # feat: (B, C, H, W) or (C, H, W)
        if feat.dim() == 4:
            feat = feat[0]
        fmap = feat.mean(0).numpy()  # Average over channels
        fmap = np.maximum(fmap, 0)
        fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-8)
        fmap = cv2.resize(fmap, (img_shape[1], img_shape[0]))
        return fmap

    # Generate heatmaps
    rgb_heatmap = featuremap_to_heatmap(activations['rgb'], rgb_image.shape)
    chm_heatmap = featuremap_to_heatmap(activations['chm'], rgb_image.shape)
    hcf_heatmap = featuremap_to_heatmap(activations['hcf'], rgb_image.shape)

    # Overlay and plot
    def overlay_heatmap(img, heatmap, alpha=0.5, title='', save_path=None):
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        # overlay = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)
        plt.figure(figsize=(8, 8))
        # plt.imshow(overlay)
        plt.imshow(heatmap_color)
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
        plt.show()

    # Convert RGB image to uint8 if needed
    rgb_disp = rgb_image
    if rgb_disp.dtype != np.uint8:
        rgb_disp = (rgb_disp * 255).astype(np.uint8)

    overlay_heatmap(rgb_disp, rgb_heatmap, alpha, 'Bộ đặc trưng RGB', f"./inferences/{img_id}-rgb-feature.png")
    overlay_heatmap(rgb_disp, chm_heatmap, alpha, 'Bộ đặc trưng CHM', f"./inferences/{img_id}-chm-feature.png")
    overlay_heatmap(rgb_disp, hcf_heatmap, alpha, 'Bộ đặc trưng kết hợp', f"./inferences/{img_id}-fused-feature.png")

    return rgb_heatmap, chm_heatmap, hcf_heatmap

def visualize_chm_peak_intervals(tif_path, sigma=3, prominence=5):
    """Paper-style interval detection for CHM."""
    # Load CHM and preprocess
    with rasterio.open(tif_path) as src:
        chm = src.read(1).astype(float)
        profile = src.profile

    # Print value range of chm
    print(f"CHM Value Range: {np.nanmin(chm)} - {np.nanmax(chm)}")
    
    # Normalize to 0-255 and remove invalid values
    chm = np.where(chm == src.nodata, np.nan, chm)
    chm_normalized = (chm - np.nanmin(chm)) / (np.nanmax(chm) - np.nanmin(chm)) * 255
    chm_normalized = chm_normalized.astype(np.uint8)

    # Compute histogram
    hist, bin_edges = np.histogram(chm_normalized[~np.isnan(chm_normalized)], bins=256, range=(0, 255))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Smooth histogram
    smoothed_hist = gaussian_filter(hist, sigma=sigma)
    # smoothed_hist = hist

    # Detect peaks
    peaks, _ = find_peaks(smoothed_hist, prominence=prominence)
    peaks = sorted(peaks, key=lambda x: smoothed_hist[x], reverse=True)  # Sort by height

    # Find intervals around peaks
    intervals = []
    for p in peaks:
        left, right = expand_interval(smoothed_hist, p)
        intervals.append((bin_edges[left], bin_edges[right + 1]))  # +1 to include right edge

    # Merge overlapping intervals (simple approach)
    intervals.sort()
    merged = []
    for interval in intervals:
        if not merged:
            merged.append(interval)
        else:
            last = merged[-1]
            if interval[0] <= last[1]:
                merged[-1] = (last[0], max(last[1], interval[1]))
            else:
                merged.append(interval)
    
    # Background = entire range minus merged intervals
    background = [(0, 255)]
    for interval in merged:
        background = [(start, end) for bg in background for (start, end) in 
                      [(bg[0], interval[0]), (interval[1], bg[1])] if start < end]
    intervals = merged + background

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(intervals)))

    # Plot histogram with intervals
    for i, (left, right) in enumerate(intervals):
        mask = (bin_centers >= left) & (bin_centers < right)
        ax1.bar(bin_centers[mask], hist[mask], 
                width=bin_edges[1] - bin_edges[0], 
                color=colors[i], alpha=0.7)
    ax1.plot(bin_centers, smoothed_hist, 'r--', lw=1, label='Smoothed')
    ax1.scatter(bin_centers[peaks], smoothed_hist[peaks], c='red', marker='x', label='Peaks')
    ax1.legend()
    ax1.set_title("CHM Histogram with Dynamically Expanded Intervals")

    # Plot CHM with colored intervals
    chm_rgb = np.zeros((*chm.shape, 3))
    for i, (left, right) in enumerate(intervals[:-1]):  # Exclude background
        mask = (chm_normalized >= left) & (chm_normalized < right)
        chm_rgb[mask] = colors[i][:3]
    ax2.imshow(chm_rgb)
    ax2.set_title("CHM with Interval Highlighting")
    ax2.axis('off')

    plt.tight_layout()
    plt.show()

def expand_interval(smoothed_hist, peak_idx):
    """Find left/right boundaries where the histogram starts increasing."""
    # Expand left from peak
    left = peak_idx
    while left > 0 and smoothed_hist[left - 1] < smoothed_hist[left]:
        left -= 1
    
    # Expand right from peak
    right = peak_idx
    while right < len(smoothed_hist) - 1 and smoothed_hist[right + 1] < smoothed_hist[right]:
        right += 1
    
    return left, right

def equalize_chm(tif_path, method='global', clip_limit=0.03):
    """
    Args:
        tif_path: Path to CHM GeoTIFF
        method: 'global' (standard equalization) or 'adaptive' (CLAHE)
        clip_limit: For CLAHE (higher = more contrast, but may amplify noise)
    """
    # Load CHM and preprocess
    with rasterio.open(tif_path) as src:
        chm = src.read(1).astype(float)
        profile = src.profile.copy()
    
    # Mask invalid values (NoData)
    chm = np.where(chm == profile['nodata'], np.nan, chm)
    valid_pixels = chm[~np.isnan(chm)]

    # Normalize CHM to 0–255 (required for histogram equalization)
    chm_norm = (valid_pixels - np.nanmin(valid_pixels)) / \
               (np.nanmax(valid_pixels) - np.nanmin(valid_pixels)) * 255
    chm_norm = chm_norm.astype(np.uint8)

    # Apply histogram equalization
    if method == 'global':
        chm_eq = exposure.equalize_hist(chm_norm) * 255
    elif method == 'adaptive':
        clahe = exposure.equalize_adapthist(chm_norm, clip_limit=clip_limit)
        chm_eq = clahe * 255
    chm_eq = chm_eq.astype(np.uint8)

    # Reconstruct the equalized CHM (preserve NaNs)
    chm_equalized = np.full_like(chm, np.nan)
    chm_equalized[~np.isnan(chm)] = chm_eq

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Original CHM
    axes[0, 0].imshow(chm, cmap='viridis', vmin=np.nanmin(chm), vmax=np.nanmax(chm))
    axes[0, 0].set_title("Original CHM")
    axes[0, 0].axis('off')
    
    # Equalized CHM
    axes[0, 1].imshow(chm_equalized, cmap='viridis', vmin=0, vmax=255)
    axes[0, 1].set_title(f"{method.capitalize()} Equalized CHM")
    axes[0, 1].axis('off')
    
    # Original histogram
    axes[1, 0].hist(chm_norm.flatten(), bins=256, range=(0, 255), color='blue', alpha=0.7)
    axes[1, 0].set_title("Original CHM Histogram")
    
    # Equalized histogram
    axes[1, 1].hist(chm_eq.flatten(), bins=256, range=(0, 255), color='red', alpha=0.7)
    axes[1, 1].set_title(f"{method.capitalize()} Equalized Histogram")
    
    plt.tight_layout()
    plt.show()

def visualize_equalized_chm_with_annotations(dataset, img_id, method='adaptive', clip_limit=0.03):
    """
    Visualize RGB image with annotations, equalized CHM, and histograms.
    
    Args:
        dataset: Your TreeDataset instance
        img_id: Image ID to visualize
        method: 'global' or 'adaptive' histogram equalization
        clip_limit: For CLAHE (only used if method='adaptive')
    """
    # Load data
    rgb_image, chm_image = dataset.get_image(img_id)
    target = dataset.get_target_with_id(img_id)
    
    # Equalize CHM
    # with rasterio.open(dataset.get_chm_path(img_id)) as src:
    #     chm = src.read(1).astype(float)
    #     profile = src.profile.copy()
    
    # Mask invalid values and equalize
    # chm = np.where(chm == profile['nodata'], np.nan, chm)
    # valid_pixels = chm[~np.isnan(chm)]
    valid_pixels = chm_image[~np.isnan(chm_image)]
    chm_norm = (valid_pixels - np.nanmin(valid_pixels)) / \
               (np.nanmax(valid_pixels) - np.nanmin(valid_pixels)) * 255
    chm_norm = chm_norm.astype(np.uint8)
    
    if method == 'global':
        chm_eq = exposure.equalize_hist(chm_norm) * 255
    elif method == 'adaptive':
        clahe = exposure.equalize_adapthist(chm_norm, clip_limit=clip_limit)
        chm_eq = clahe * 255
    chm_eq = chm_eq.astype(np.uint8)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(18, 12))
    
    # RGB Image with Annotations
    axes[0].imshow(rgb_image)
    for bbox in target['boxes']:
        rect = patches.Rectangle(
            (bbox[0], bbox[1]),
            bbox[2] - bbox[0],
            bbox[3] - bbox[1],
            linewidth=2,
            edgecolor='r',
            facecolor='none'
        )
        axes[0].add_patch(rect)
    axes[0].set_title('RGB Image with Annotations')
    axes[0].axis('off')
    
    # Equalized CHM
    chm_display = np.full_like(chm_image, np.nan)
    chm_display[~np.isnan(chm_image)] = chm_eq
    axes[1].imshow(chm_image, cmap='viridis')
    axes[1].set_title(f'{method.capitalize()} Equalized CHM')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()

def visualize_rgb_chm_intervals(dataset, img_id, sigma=3, prominence=5):
    """
    Visualize RGB image with annotations and CHM with interval highlighting.
    
    Args:
        dataset: Your TreeDataset instance
        img_id: Image ID to visualize
        sigma: Gaussian filter strength for histogram smoothing
        prominence: Minimum peak prominence for interval detection
    """
    # Load data
    rgb_image, chm_image = dataset.get_image(img_id)
    target = dataset.get_target_with_id(img_id)

    # Print RGB image value range
    print(f"RGB Image Value Range: {np.nanmin(rgb_image)} - {np.nanmax(rgb_image)}")
     
    # CHM preprocessing
    chm = chm_image
    chm_normalized = (chm - np.nanmin(chm)) / (np.nanmax(chm) - np.nanmin(chm)) * 255
    chm_normalized = chm_normalized.astype(np.uint8)
    
    # Compute histogram and intervals (using existing logic)
    hist, bin_edges = np.histogram(chm_normalized[~np.isnan(chm_normalized)], bins=256, range=(0, 255))
    # bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    smoothed_hist = gaussian_filter(hist, sigma=sigma)
    peaks, _ = find_peaks(smoothed_hist, prominence=prominence)
    peaks = sorted(peaks, key=lambda x: smoothed_hist[x], reverse=True)
    
    # # Calculate intervals
    # intervals = []
    # for p in peaks:
    #     left, right = expand_interval(smoothed_hist, p)
    #     intervals.append((bin_edges[left], bin_edges[right + 1]))
    
    # # Merge intervals and create background (existing logic)
    # # ... [include the interval merging code from your existing function] ...
    # intervals.sort()
    # merged = []
    # for interval in intervals:
    #     if not merged:
    #         merged.append(interval)
    #     else:
    #         last = merged[-1]
    #         if interval[0] <= last[1]:
    #             merged[-1] = (last[0], max(last[1], interval[1]))
    #         else:
    #             merged.append(interval)

    intervals = merge_intervals_with_valleys(smoothed_hist, peaks, bin_edges)
    merged = intervals.copy()
    
    # Background = entire range minus merged intervals
    background = [(0, 255)]
    for interval in merged:
        background = [(start, end) for bg in background for (start, end) in 
                      [(bg[0], interval[0]), (interval[1], bg[1])] if start < end]
    intervals = merged + background
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # RGB Image with Annotations
    ax1.imshow(rgb_image)
    for bbox in target['boxes']:
        rect = patches.Rectangle(
            (bbox[0], bbox[1]),
            bbox[2] - bbox[0],
            bbox[3] - bbox[1],
            linewidth=2,
            edgecolor='r',
            facecolor='none'
        )
        ax1.add_patch(rect)
    ax1.set_title(f'RGB Image with Annotations ({img_id})')
    ax1.axis('off')
    
    # CHM with Interval Highlighting
    chm_rgb = np.zeros((*chm_image.shape, 3))
    colors = plt.cm.viridis(np.linspace(0, 1, len(intervals)))
    for i, (left, right) in enumerate(intervals[:-1]):  # Exclude background
        mask = (chm_normalized >= left) & (chm_normalized < right)
        chm_rgb[mask] = colors[i][:3]
    ax2.imshow(chm_rgb)
    ax2.set_title('CHM with Height Intervals')
    ax2.axis('off')
    
    # Add legend for CHM intervals
    legend_elements = [patches.Patch(color=colors[i], 
                      label=f'{intervals[i][0]:.1f}-{intervals[i][1]:.1f}') 
                     for i in range(len(intervals)-1)]
    ax2.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()

def find_valleys(smoothed_hist):
    """Find valleys (local minima) between peaks by inverting the histogram."""
    inverted_hist = -smoothed_hist
    valleys, _ = find_peaks(inverted_hist, prominence=1)  # Adjust prominence as needed
    return valleys

def merge_intervals_with_valleys(smoothed_hist, peaks, bin_edges):
    """Use valleys between peaks as interval boundaries."""
    valleys = find_valleys(smoothed_hist)
    
    # Add boundary valleys (start=0, end=255) if needed
    valleys = np.concatenate(([0], valleys, [len(smoothed_hist)-1]))
    
    # Pair peaks with neighboring valleys
    intervals = []
    for peak in peaks:
        # Find left valley (last valley < peak)
        left_valley = valleys[valleys < peak][-1] if np.any(valleys < peak) else 0
        
        # Find right valley (first valley > peak)
        right_valley = valleys[valleys > peak][0] if np.any(valleys > peak) else len(smoothed_hist)-1
        
        intervals.append((
            bin_edges[left_valley], 
            bin_edges[right_valley + 1]  # +1 to include right edge
        ))
    
    # Merge only truly overlapping intervals (rare now)
    merged = []
    for interval in sorted(intervals):
        if not merged:
            merged.append(interval)
        else:
            last = merged[-1]
            if interval[0] < last[1]:  # Overlap only if start < previous end
                merged[-1] = (last[0], max(last[1], interval[1]))
            else:
                merged.append(interval)
    
    return merged

def segment_chm_peaks(raw_chm, min_distance=5, min_height=5, sigma=1.5):
    """
    Segment tree crowns in CHM using watershed segmentation.
    
    Args:
        raw_chm: CHM array from dataset.get_image()
        min_distance: Minimum distance between peaks (pixels)
        min_height: Minimum height threshold for peak detection
        sigma: Gaussian smoothing strength
    
    Returns:
        coordinates: Peak locations (y, x)
        labels: Watershed segmentation labels
        chm_smoothed: Smoothed CHM array
    """
    # Convert to float and smooth
    chm = raw_chm.astype(float)
    chm_smoothed = filters.gaussian(chm, sigma=sigma)

    # Detect local maxima (tree tops)
    coordinates = peak_local_max(
        chm_smoothed,
        min_distance=min_distance,
        threshold_abs=min_height,
        exclude_border=False
    )

    # Create markers array
    markers = np.zeros_like(chm_smoothed, dtype=bool)
    markers[tuple(coordinates.T)] = True
    markers = ndi.label(markers)[0]

    # Watershed segmentation
    elevation_map = -chm_smoothed  # Invert for watershed
    labels = watershed(
        elevation_map,
        markers,
        mask=chm_smoothed > min_height,
        compactness=0.01
    )

    # Post-processing
    labels = morphology.remove_small_objects(labels, min_size=20)
    
    return coordinates, labels, chm_smoothed

def visualize_chm_segmentation(dataset, img_id, min_distance=5, min_height=5):
    """
    Visualize RGB annotations with CHM segmentation results.
    """
    # Get data from dataset
    rgb_image, raw_chm = dataset.get_image(img_id)
    target = dataset.get_target_with_id(img_id)

    # Perform segmentation
    peaks, labels, chm_smoothed = segment_chm_peaks(raw_chm, min_distance, min_height)

    # Create visualization
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 7))

    # RGB with Annotations
    ax1.imshow(rgb_image)
    for bbox in target['boxes']:
        rect = patches.Rectangle(
            (bbox[0], bbox[1]),
            bbox[2] - bbox[0],
            bbox[3] - bbox[1],
            linewidth=1.5,
            edgecolor='r',
            facecolor='none'
        )
        ax1.add_patch(rect)
    ax1.set_title(f'RGB with Annotations ({img_id})')
    ax1.axis('off')

    # CHM with Detected Peaks
    ax2.imshow(chm_smoothed, cmap='viridis')
    ax2.scatter(peaks[:, 1], peaks[:, 0], 
                c='red', s=10, marker='x', 
                label='Tree Tops')
    ax2.set_title('CHM with Detected Peaks')
    ax2.axis('off')

    # Watershed Segmentation Result
    ax3.imshow(labels, cmap='tab20')
    ax3.set_title('Watershed Segmentation')
    ax3.axis('off')

    plt.tight_layout()
    plt.show()

def segment_in_bboxes(rgb_image, raw_chm, boxes, min_height=5, min_distance=5, sigma=1.5):
    """
    Segment tree crowns within bounding boxes using CHM data.
    
    Args:
        rgb_image: RGB image array (H, W, 3)
        raw_chm: CHM array (H, W)
        boxes: List of bounding boxes in format [[xmin, ymin, xmax, ymax], ...]
        min_height: Minimum tree height threshold
        min_distance: Minimum distance between tree peaks
        sigma: Gaussian smoothing strength
    
    Returns:
        full_mask: Combined segmentation mask for all boxes
        chm_smoothed: Smoothed CHM array
    """
    # Preprocess CHM
    chm = raw_chm.astype(float)
    chm_smoothed = filters.gaussian(chm, sigma=sigma)
    full_mask = np.zeros_like(chm, dtype=bool)

    for box in boxes:
        # Convert box coordinates to integers
        xmin, ymin, xmax, ymax = map(int, box)
        
        # Extract CHM patch
        chm_patch = chm_smoothed[ymin:ymax, xmin:xmax]
        
        # Skip empty patches
        if chm_patch.size == 0:
            continue

        # Detect local maxima within patch
        coordinates = peak_local_max(
            chm_patch,
            min_distance=min_distance,
            threshold_abs=min_height,
            exclude_border=False
        )

        if len(coordinates) == 0:
            continue

        # Create markers
        markers = np.zeros_like(chm_patch, dtype=bool)
        markers[tuple(coordinates.T)] = True
        markers = ndi.label(markers)[0]

        # Watershed segmentation
        elevation_map = -chm_patch
        labels = watershed(
            elevation_map,
            markers,
            mask=chm_patch > min_height,
            compactness=0.01
        )

        # Post-process and add to full mask
        labels = morphology.remove_small_objects(labels, min_size=20)
        full_mask[ymin:ymax, xmin:xmax] |= labels > 0

    return full_mask, chm_smoothed

def visualize_segmented_crowns(dataset, img_id, min_height=5, min_distance=5):
    """
    Visualize RGB boxes with CHM segmentation results.
    """
    # Get data from dataset
    rgb_image, raw_chm = dataset.get_image(img_id)
    target = dataset.get_target_with_id(img_id)
    boxes = target['boxes'].cpu().numpy()

    # Perform segmentation
    mask, chm_smoothed = segment_in_bboxes(rgb_image, raw_chm, boxes, 
                                         min_height=min_height, 
                                         min_distance=min_distance)

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))

    # RGB image with bounding boxes
    ax1.imshow(rgb_image)
    for box in boxes:
        xmin, ymin, xmax, ymax = box
        rect = patches.Rectangle(
            (xmin, ymin), xmax-xmin, ymax-ymin,
            linewidth=1.5, edgecolor='r', facecolor='none'
        )
        ax1.add_patch(rect)
    ax1.set_title(f'RGB with Bounding Boxes ({img_id})')
    ax1.axis('off')

    # CHM with segmentation overlay
    ax2.imshow(chm_smoothed, cmap='gray')
    ax2.imshow(np.ma.masked_where(~mask, mask), 
              cmap='autumn', alpha=0.5, 
              vmin=0, vmax=1)
    ax2.set_title('CHM with Segmented Tree Crowns')
    ax2.axis('off')

    plt.tight_layout()
    plt.show()

def enhance_full_chm(raw_chm, sigma=2, min_size=50):
    """Enhance full CHM using watershed segmentation masks
    
    Args:
        raw_chm: Original CHM array (H, W)
        sigma: Gaussian smoothing for segmentation
        min_size: Minimum crown region size (pixels)
        
    Returns:
        enhanced_chm: CHM with contrast-enhanced crowns
        labels: Full segmentation mask
    """
    # Segment entire CHM
    _, labels, _ = segment_chm_peaks(raw_chm, sigma=sigma)
    
    # Enhance each region
    enhanced_chm = np.zeros_like(raw_chm)
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        if label == 0:  # Skip background
            continue
            
        mask = labels == label
        if np.sum(mask) < min_size:
            continue
            
        # Extract and enhance region
        region = raw_chm[mask]
        enhanced_region = exposure.equalize_hist(region)
        
        # Rescale to original range
        enhanced_region = exposure.rescale_intensity(
            enhanced_region, 
            in_range=(enhanced_region.min(), enhanced_region.max()),
            out_range=(raw_chm.min(), raw_chm.max())
        )
        
        # Smooth edges
        expanded_mask = morphology.dilation(mask, morphology.disk(2))
        enhanced_chm[mask] = 0.6*enhanced_region + 0.4*raw_chm[mask]
        
    return enhanced_chm, labels

def visualize_full_enhancement(dataset, img_id):
    """Full visualization pipeline"""
    # Load data
    rgb_image, raw_chm = dataset.get_image(img_id)
    target = dataset.get_target_with_id(img_id)
    
    # Process CHM
    enhanced_chm, labels = enhance_full_chm(raw_chm)
    
    # Visualization
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
    
    # RGB with annotations
    ax1.imshow(rgb_image)
    for box in target['boxes'].cpu().numpy():
        xmin, ymin, xmax, ymax = box
        rect = plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, 
                           linewidth=1.5, edgecolor='r', facecolor='none')
        ax1.add_patch(rect)
    ax1.set_title(f'RGB Annotations ({img_id})')
    ax1.axis('off')
    
    # Original CHM
    ax2.imshow(raw_chm, cmap='viridis', vmin=raw_chm.min(), vmax=raw_chm.max())
    ax2.set_title('Original CHM')
    ax2.axis('off')
    
    # Enhanced CHM
    ax3.imshow(enhanced_chm, cmap='viridis', vmin=raw_chm.min(), vmax=raw_chm.max())
    ax3.set_title('Enhanced Crown Regions')
    ax3.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    data_dir = 'D:/Self_Practicing/Computer Vision/research/Experiments/src/data/preprocessed/'
    # # data_dir = '/media02/lqngoc22/thesis-tree-delineation/data/preprocessed'
    dataset = TreeDataset(data_dir, 2, train=False)
    # visualize_image_with_annotation(dataset, 'TEAK_043_2018', mode='chm', save_path='./inferences')

    # visualize_chm_peak_intervals('D:/Self_Practicing/Computer Vision/research/Experiments/src/data/preprocessed/test/CHM/MLBS_073_2018_chm.tif', sigma=3, prominence=5)
    # equalize_chm('D:/Self_Practicing/Computer Vision/research/Experiments/src/data/preprocessed/test/CHM/MLBS_071_2018_chm.tif', method='adaptive', clip_limit=0.03)
    # visualize_equalized_chm_with_annotations(dataset, 'MLBS_069_2018', method='adaptive', clip_limit=0.03)
    # visualize_rgb_chm_intervals(dataset, 'MLBS_069_2018', sigma=3, prominence=5)
    # visualize_chm_segmentation(dataset, '2018_TEAK_3_322000_4103000_image_50', min_distance=10, min_height=10)
    # visualize_segmented_crowns(dataset, '2018_TEAK_3_322000_4103000_image_50', min_distance=3, min_height=20)
    # visualize_full_enhancement(dataset, 'MLBS_069_2018')

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yml", help="path to the config file")
    parser.add_argument("--img-id", required=True, help="ID of the image to visualize")
    parser.add_argument("--save-path", default="inferences", help="path to save the visualized image")
    parser.add_argument("--conf-threshold", type=float, default=0.5, help="Confidence threshold for filtering predictions")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold for non-maximum suppression")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() and config['test']['use_cuda'] else "cpu")
    data_dir = config['test']['data_dir'] 
    dataset = TreeDataset(data_dir, config['test']['max_workers'], config['test']['verbose'], train=False)
    
    num_classes = max(dataset.classes) + 1
    if config['test']['model'] == 'faster_rcnn':
        model = models.fasterrcnn_fpn_resnet50(pretrained=True, num_classes=num_classes).to(device)
    elif config['test']['model'] == 'hcf_rcnn':
        model = models.fused_rcnn_resnet50(pretrained=True, num_classes=num_classes, backbone_name='hcf_resnet50').to(device)
    elif config['test']['model'] == 'deepforest_retinanet':
        model = DeepForestWrapper(model_type='retinanet').to(device)

    if os.path.exists(config['test']['ckpt_path']) and not config['test']['model'].lower().startswith('deepforest'):
        checkpoint = torch.load(config['test']['ckpt_path'], map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model"]) 
        model_state_dict = model.state_dict()
        # for i, name in enumerate(model_state_dict):
        #     print(name)

    img_id = args.img_id
    # save_path = os.path.join(args.save_path, f"{img_id}-{config['test']['model']}.png")
    # visualize_image_with_prediction(dataset, img_id, model, device, save_path, args.conf_threshold, args.iou_threshold, config=config)

    save_path = os.path.join(args.save_path, f"{img_id}-{config['test']['model']}-attention")
    visualize_hcf_fusion_attention(dataset, img_id, model, device)
