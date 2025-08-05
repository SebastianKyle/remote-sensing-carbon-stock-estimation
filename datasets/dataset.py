import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
import random
from torchvision import transforms
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import rasterio as rio
from rasterio.errors import NotGeoreferencedWarning
import json
import warnings

# from skimage import io
# from skimage import exposure
# from scipy.ndimage import gaussian_filter
# from scipy.signal import find_peaks
# import rasterio
# from skimage import filters, morphology, measure
# from skimage.segmentation import watershed
# from skimage.feature import peak_local_max
# from scipy import ndimage as ndi
# import matplotlib.patches as patches

class TreeDataset(Dataset):
    """
    Main class for Tree Dataset.
    """

    def __init__(self, data_dir, max_workers=2, verbose=False, train=True, use_edge=False):
        self.data_dir = data_dir
        self.train_dir = os.path.join(data_dir, 'train')
        self.test_dir = os.path.join(data_dir, 'test')
        self.max_workers = max_workers
        self.verbose = verbose
        self.ids = []
        self.imgs = {}
        self.classes = {1: "Tree"} # Tree
        self.train = train

        # Remove the random transforms from Compose and store them separately
        self.color_jitter = ColorJitter(hue=0.15, saturation=0.25, brightness=0.15) if train else None
        self.do_hflip = True if train else False
        self.do_vflip = True if train else False

        self.load_data()
        
    def __getitem__(self, idx):
        img_id = self.ids[idx]
        rgb_image, chm_image = self.get_image(img_id)
        rgb_image = transforms.ToTensor()(rgb_image)
        chm_image = transforms.ToTensor()(chm_image)

        target = self.get_target(idx)

        # Apply augmentations during training
        if self.train:
            # Use the same random decision for both images
            if self.do_hflip and random.random() < 0.5:
                rgb_image = F.hflip(rgb_image)
                chm_image = F.hflip(chm_image)
                if target['boxes'].numel() > 0:
                    target['boxes'][:, [0, 2]] = chm_image.shape[-1] - target['boxes'][:, [2, 0]]

            if self.do_vflip and random.random() < 0.5:
                rgb_image = F.vflip(rgb_image)
                chm_image = F.vflip(chm_image)
                if target['boxes'].numel() > 0:
                    target['boxes'][:, [1, 3]] = chm_image.shape[-2] - target['boxes'][:, [3, 1]]

            # Apply color jitter only to RGB image
            if self.color_jitter is not None:
                rgb_image = self.color_jitter(rgb_image)

        return rgb_image, chm_image, target 

    def __len__(self):
        return len(self.ids)
    
    def get_image(self, img_id):
        img_info = self.imgs[img_id]

        warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
        try:
            with rio.open(img_info['rgb_path']) as src:
                rgb_image = src.read([1, 2, 3]).transpose(1, 2, 0)
            
            with rio.open(img_info['chm_path']) as src:
                chm_image = src.read(1)

            # enhanced_chm, _ = enhance_full_chm(chm_image)
            # # Stack chm and enhanced chm to form 3 channels (chm - chm - enhanced chm)
            # chm_image = np.stack((chm_image, enhanced_chm, chm_image), axis=-1)
            
            return rgb_image, chm_image
        except FileNotFoundError:
            print(f"Image {img_id} not found, skipping.")
            return None, None

    def get_target(self, idx):
        img_id = self.ids[idx]
        img_info = self.imgs[img_id]
        ann_path = img_info['ann_path']
        
        with open(ann_path, 'r') as f:
            data = json.load(f)
            
        bounding_boxes = []
        labels = []
        for obj in data['annotations']:
            if isinstance(obj, dict) and 'bndbox' in obj:
                bbox = obj['bndbox']
                bounding_boxes.append([
                    bbox['xmin'] if isinstance(bbox['xmin'], int) else int(bbox['xmin']),
                    bbox['ymin'] if isinstance(bbox['ymin'], int) else int(bbox['ymin']),
                    bbox['xmax'] if isinstance(bbox['xmax'], int) else int(bbox['xmax']),
                    bbox['ymax'] if isinstance(bbox['ymax'], int) else int(bbox['ymax']),
                ])
                labels.append(1) # Tree
                # labels.append(obj['name'])

        bounding_boxes = torch.tensor(bounding_boxes, dtype=torch.float32)
        labels = torch.tensor(labels)
        
        target = dict(image_id=torch.tensor([idx]), boxes=bounding_boxes, labels=labels)

        return target

    def get_target_with_id(self, img_id):
        img_info = self.imgs[img_id]
        ann_path = img_info['ann_path']
        
        with open(ann_path, 'r') as f:
            data = json.load(f)
            
        bounding_boxes = []
        labels = []
        for obj in data['annotations']:
            if isinstance(obj, dict) and 'bndbox' in obj:
                bbox = obj['bndbox']
                bounding_boxes.append([
                    bbox['xmin'] if isinstance(bbox['xmin'], int) else int(bbox['xmin']),
                    bbox['ymin'] if isinstance(bbox['ymin'], int) else int(bbox['ymin']),
                    bbox['xmax'] if isinstance(bbox['xmax'], int) else int(bbox['xmax']),
                    bbox['ymax'] if isinstance(bbox['ymax'], int) else int(bbox['ymax']),
                ])
                labels.append(1) # Tree

        bounding_boxes = torch.tensor(bounding_boxes, dtype=torch.float32)
        labels = torch.tensor(labels)
        
        target = dict(boxes=bounding_boxes, labels=labels)

        return target

    def load_data(self):
        annotation_dir = os.path.join(self.train_dir, 'annotations') if self.train else os.path.join(self.test_dir, 'annotations')

        for ann_file in os.listdir(annotation_dir):
            if ann_file.endswith('.json'):
                # img_id = os.path.splitext(ann_file)[0] 
                # self.ids.append(img_id)
                
                ann_path = os.path.join(annotation_dir, ann_file)
                with open(ann_path, 'r') as f:
                    data = json.load(f)

                img_id = data['id']
                self.ids.append(img_id)
                    
                rgb_path = data['rgb_path']
                chm_path = data['chm_path']

                self.imgs[img_id] = {
                    'rgb_path': rgb_path,
                    'chm_path': chm_path,
                    'ann_path': ann_path,
                }

class ColorJitter:
    def __init__(self, hue=0.15, saturation=0.25, brightness=0.15):
        self.hue = hue
        self.saturation = saturation
        self.brightness = brightness

    def __call__(self, img):
        # Randomly apply color jittering
        if random.random() < 0.5:
            img = F.adjust_hue(img, random.uniform(-self.hue, self.hue))
        if random.random() < 0.5:
            img = F.adjust_saturation(img, random.uniform(1-self.saturation, 1+self.saturation))
        if random.random() < 0.5:
            img = F.adjust_brightness(img, random.uniform(1-self.brightness, 1+self.brightness))
        return img

# def segment_chm_peaks(raw_chm, min_distance=5, min_height=5, sigma=1.5):
#     """
#     Segment tree crowns in CHM using watershed segmentation.
    
#     Args:
#         raw_chm: CHM array from dataset.get_image()
#         min_distance: Minimum distance between peaks (pixels)
#         min_height: Minimum height threshold for peak detection
#         sigma: Gaussian smoothing strength
    
#     Returns:
#         coordinates: Peak locations (y, x)
#         labels: Watershed segmentation labels
#         chm_smoothed: Smoothed CHM array
#     """
#     # Convert to float and smooth
#     chm = raw_chm.astype(float)
#     chm_smoothed = filters.gaussian(chm, sigma=sigma)

#     # Detect local maxima (tree tops)
#     coordinates = peak_local_max(
#         chm_smoothed,
#         min_distance=min_distance,
#         threshold_abs=min_height,
#         exclude_border=False
#     )

#     # Create markers array
#     markers = np.zeros_like(chm_smoothed, dtype=bool)
#     markers[tuple(coordinates.T)] = True
#     markers = ndi.label(markers)[0]

#     # Watershed segmentation
#     elevation_map = -chm_smoothed  # Invert for watershed
#     labels = watershed(
#         elevation_map,
#         markers,
#         mask=chm_smoothed > min_height,
#         compactness=0.01
#     )

#     # Post-processing
#     warnings.filterwarnings("ignore", category=UserWarning)
#     labels = morphology.remove_small_objects(labels, min_size=20)
    
#     return coordinates, labels, chm_smoothed

# def enhance_full_chm(raw_chm, sigma=2, min_size=50):
#     """Enhance full CHM using watershed segmentation masks
    
#     Args:
#         raw_chm: Original CHM array (H, W)
#         sigma: Gaussian smoothing for segmentation
#         min_size: Minimum crown region size (pixels)
        
#     Returns:
#         enhanced_chm: CHM with contrast-enhanced crowns
#         labels: Full segmentation mask
#     """
#     # Segment entire CHM
#     _, labels, _ = segment_chm_peaks(raw_chm, sigma=sigma)
    
#     # Enhance each region
#     enhanced_chm = np.zeros_like(raw_chm)
#     unique_labels = np.unique(labels)
    
#     for label in unique_labels:
#         if label == 0:  # Skip background
#             continue
            
#         mask = labels == label
#         if np.sum(mask) < min_size:
#             continue
            
#         # Extract and enhance region
#         region = raw_chm[mask]
#         enhanced_region = exposure.equalize_hist(region)
        
#         # Rescale to original range
#         enhanced_region = exposure.rescale_intensity(
#             enhanced_region, 
#             in_range=(enhanced_region.min(), enhanced_region.max()),
#             out_range=(raw_chm.min(), raw_chm.max())
#         )
        
#         # Smooth edges
#         expanded_mask = morphology.dilation(mask, morphology.disk(2))
#         enhanced_chm[mask] = 0.6*enhanced_region + 0.4*raw_chm[mask]
        
#     return enhanced_chm, labels