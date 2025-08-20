import os
import yaml
import argparse
import numpy as np
import rasterio
from joblib import load
import torch
from torchvision.ops import nms
from torchvision import transforms
from carbon_stock_estimation.utils import split_image_with_offsets, map_boxes_to_large, merge_detections, merge_overlapping_boxes, extract_features_for_box, visualize_detections, plot_carbon_distribution, visualize_with_latlon
import models
import cv2
import pandas as pd

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

# --- Main pipeline ---
def estimate_carbon_from_large_tile(rgb_path, chm_path, hsi_path, model, rf_model_path, tile_size=400, device='cuda', aop_bands_path='./neon_aop_bands.csv'):
    # 1. Read large images
    with rasterio.open(rgb_path) as src:
        rgb_large = src.read([1, 2, 3])  # (3, H, W)
        transform = src.transform
        crs = src.crs
        bounds = src.bounds  # Bounding box in real-world coordinates
        width = src.width
        height = src.height
        resolution = (transform[0], -transform[4])  # pixel size (x, y)
    with rasterio.open(chm_path) as src:
        chm_large = src.read(1)  # (H, W)
    with rasterio.open(hsi_path) as src:
        hsi_large = src.read()
        hsi_large = np.transpose(hsi_large, (1, 2, 0))  # (H, W, bands)
    hsi_large = np.where(hsi_large == -9999, 2, hsi_large)

    area_ha = (width * resolution[0]) * (height * resolution[1]) / 10_000

    scale_rgb_to_chm = rgb_large.shape[1] / chm_large.shape[0]
    scale_rgb_to_hsi = rgb_large.shape[1] / hsi_large.shape[0]

    chm_large = np.clip(chm_large, np.percentile(chm_large, 3), np.percentile(chm_large, 99))
    chm_large_scaled = (chm_large - np.min(chm_large)) / (np.max(chm_large) - np.min(chm_large))
    chm_large_scaled = (chm_large_scaled * 255).astype(np.uint8)
    chm_large_resized = cv2.resize(chm_large_scaled, (rgb_large.shape[2], rgb_large.shape[1]), interpolation=cv2.INTER_CUBIC)

    # 2. Split into crops
    rgb_tiles, offsets = split_image_with_offsets(rgb_large, (tile_size, tile_size))
    chm_tiles, _ = split_image_with_offsets(chm_large, (tile_size, tile_size), scale_factor=0.1)
    # (HSI splitting only needed for feature extraction)

    # 3. Run detection on each tile
    all_boxes = []
    all_scores = []
    for rgb_tile, chm_tile, offset in zip(rgb_tiles, chm_tiles, offsets):
        # Prepare input for model (adjust as needed for your model)
        rgb_tensor = transforms.ToTensor()(rgb_tile.transpose(1, 2, 0)).to(device)  # (3, 400, 400)
        chm_tensor = transforms.ToTensor()(chm_tile).to(device) # (400, 400)

        # Run detection (replace with your model's API)
        with torch.no_grad():
            detections = model([rgb_tensor], [chm_tensor])

        # detections['boxes']: (N, 4), detections['scores']: (N,)
        boxes = detections[0]['boxes'].cpu().numpy()
        scores = detections[0]['scores'].cpu().numpy()

        # Map to large image coordinates
        boxes_large = map_boxes_to_large(boxes, offset)
        all_boxes.append(boxes_large)
        all_scores.append(scores)

    # 4. Merge detections
    merged_boxes, merged_scores = merge_overlapping_boxes(all_boxes, all_scores, iou_thresh=0.2)

    # 5. Extract features for each detected crown
    features = []
    for box in merged_boxes:
        # print(box)
        x1, y1, x2, y2 = map(int, box)
        rgb_crop = rgb_large[:, y1:y2, x1:x2]

        # chm_crop = chm_large[
        #     int(y1/scale_rgb_to_chm):int(y2/scale_rgb_to_chm),
        #     int(x1/scale_rgb_to_chm):int(x2/scale_rgb_to_chm)
        # ]
        # hsi_crop = hsi_large[
        #     int(y1/scale_rgb_to_hsi):int(y2/scale_rgb_to_hsi), 
        #     int(x1/scale_rgb_to_hsi):int(x2/scale_rgb_to_hsi), 
        #     :
        # ]

        chm_x1 = int(x1 / scale_rgb_to_chm)
        chm_x2 = int(x2 / scale_rgb_to_chm)
        chm_y1 = int(y1 / scale_rgb_to_chm)
        chm_y2 = int(y2 / scale_rgb_to_chm)
        hsi_x1 = int(x1 / scale_rgb_to_hsi)
        hsi_x2 = int(x2 / scale_rgb_to_hsi)
        hsi_y1 = int(y1 / scale_rgb_to_hsi)
        hsi_y2 = int(y2 / scale_rgb_to_hsi)

        # Ensure non-empty crops
        if chm_x2 <= chm_x1 or chm_y2 <= chm_y1 or hsi_x2 <= hsi_x1 or hsi_y2 <= hsi_y1:
            print(f"Skipping degenerate crop: box={box}")
            continue

        chm_crop = chm_large[chm_y1:chm_y2, chm_x1:chm_x2]
        hsi_crop = hsi_large[hsi_y1:hsi_y2, hsi_x1:hsi_x2, :]

        if chm_crop.shape[0] == 0 or chm_crop.shape[1] == 0:
            print(f"Skipping degenerate crop: box={box}")
            continue

        # Use your feature extraction function (adapt as needed)
        feat = extract_features_for_box(rgb_crop, chm_crop, hsi_crop, aop_bands_path=aop_bands_path)
        features.append(feat)

    # 6. Predict carbon stock
    rf_model = load(rf_model_path)
    features_df = pd.DataFrame(features)
    carbon_stocks = rf_model.predict(features_df)

    # 7. Sum for total carbon stock
    total_carbon = np.sum(carbon_stocks)
    return total_carbon, merged_boxes, carbon_stocks, area_ha


def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    conf_thresh = config['inference']['conf_thresh']

    model = models.fused_rcnn_resnet50(pretrained=True, num_classes=2, box_score_thresh=conf_thresh, backbone_name='hcf_resnet50').to(device)
    checkpoint = torch.load(config['inference']['ckpt_path'], map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    rgb_tile_path = config['inference']['rgb_tile_path']
    chm_tile_path = config['inference']['chm_tile_path']
    hsi_tile_path = config['inference']['hsi_tile_path']
    rf_model_path = config['inference']['rf_model_path']

    total_carbon, merged_boxes, carbon_stocks, area_ha = estimate_carbon_from_large_tile(rgb_tile_path, chm_tile_path, hsi_tile_path, model, rf_model_path, tile_size=400, device=device, aop_bands_path=config['inference']['aop_bands_path'])

    print(f"{area_ha:.2f} ha of {config['inference']['site']} forest stores {total_carbon:.2f} kg carbon")

    # Further visualizations
    visualize_detections(rgb_tile_path, merged_boxes, carbon_stocks, total_carbon, config['inference']['site'], fig_path=config['inference']['detection_path'])
    # visualize_with_latlon(rgb_tile_path, total_carbon, config['inference']['site'], merged_boxes, save_fig=config['inference']['detection_path'])
    plot_carbon_distribution(carbon_stocks, fig_path=config['inference']['distribution_path'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Forest carbon index estimation pipeline")
    parser.add_argument('--config', type=str, default='config.yml', help='Path to carbon index estimation config file')
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    main(config)