import os
import numpy as np
from joblib import load
import torch
from torchvision.ops import nms
import cv2
import pandas as pd
from sklearn.decomposition import PCA
import rasterio
import matplotlib.pyplot as plt
from rasterio.plot import show
from rasterio.warp import transform_bounds
from pyproj import Transformer

def compute_optimal_overlap(image_size, crop_size):
    """
    Compute the minimal overlap so that the last crop aligns with the image edge.
    Returns the overlap for a given image size and crop size.
    """
    n_crops = int(np.ceil((image_size - crop_size) / crop_size)) + 1
    if n_crops == 1:
        return 0
    step = (image_size - crop_size) // (n_crops - 1)
    overlap = crop_size - step
    return overlap

# --- Helper: Split large image into overlapping 401x400 crops ---
def split_image_with_offsets(image, size=(400, 400), overlap=None, scale_factor=1):
    """
    Split image into tiles of given size and overlap.
    If scale_factor < 1, tile size and overlap are scaled down (for CHM/HSI).
    """
    h, w = image.shape[-2], image.shape[-1]
    print(f"Height, width: {h, w}")
    crop_h, crop_w = size
    if scale_factor != 1:
        crop_h = int(crop_h * scale_factor)
        crop_w = int(crop_w * scale_factor)
    # Auto-compute optimal overlap if not provided
    if overlap is None:
        overlap_h = compute_optimal_overlap(h, crop_h)
        overlap_w = compute_optimal_overlap(w, crop_w)
    else:
        overlap_h = overlap_w = int(overlap * scale_factor)
    step_h = crop_h - overlap_h
    step_w = crop_w - overlap_w
    crops = []
    offsets = []
    for y in range(0, h - crop_h + 1, step_h):
        for x in range(0, w - crop_w + 1, step_w):
            if image.ndim == 3:
                crop = image[:, y:y+crop_h, x:x+crop_w]
            else:
                crop = image[y:y+crop_h, x:x+crop_w]
            crops.append(crop)
            offsets.append((x, y))
    return crops, offsets

# --- Helper: Map detected boxes to large image coordinates ---
def map_boxes_to_large(boxes, offset):
    x_off, y_off = offset
    mapped = boxes.copy()
    mapped[:, [0, 2]] += x_off
    mapped[:, [1, 3]] += y_off
    return mapped

# --- Helper: Merge overlapping boxes using NMS ---
def merge_detections(all_boxes, all_scores, iou_thresh=0.5):
    if len(all_boxes) == 0:
        return np.zeros((0, 4)), np.zeros((0,))
    boxes = torch.tensor(np.vstack(all_boxes), dtype=torch.float32)
    scores = torch.tensor(np.hstack(all_scores), dtype=torch.float32)
    keep = nms(boxes, scores, iou_thresh)
    return boxes[keep].cpu().numpy(), scores[keep].cpu().numpy()

def merge_overlapping_boxes(boxes, scores, iou_thresh=0.5):
    import torch
    from torchvision.ops import box_iou

    boxes = torch.tensor(np.vstack(boxes), dtype=torch.float32)
    scores = torch.tensor(np.hstack(scores), dtype=torch.float32)
    keep = [True] * len(boxes)

    merged_boxes = []
    merged_scores = []

    used = set()
    for i in range(len(boxes)):
        if i in used:
            continue
        curr_box = boxes[i]
        curr_score = scores[i]
        overlaps = [i]
        for j in range(i+1, len(boxes)):
            if j in used:
                continue
            iou = box_iou(curr_box.unsqueeze(0), boxes[j].unsqueeze(0))[0, 0]
            if iou > iou_thresh:
                # Merge: take the union of the two boxes
                x1 = min(curr_box[0], boxes[j][0])
                y1 = min(curr_box[1], boxes[j][1])
                x2 = max(curr_box[2], boxes[j][2])
                y2 = max(curr_box[3], boxes[j][3])
                curr_box = torch.tensor([x1, y1, x2, y2])
                curr_score = max(curr_score, scores[j])  # or sum, or average
                used.add(j)
                overlaps.append(j)
        merged_boxes.append(curr_box.numpy())
        merged_scores.append(curr_score.item())
        used.update(overlaps)
    return np.array(merged_boxes), np.array(merged_scores)

def extract_features_for_box(rgb_crop, chm_crop, hsi_crop, aop_bands_path='./neon_aop_bands.csv'):
    bands_df = pd.read_csv(aop_bands_path)
    wavelengths = bands_df['nanometer'].values

    # RGB features: crown diameter (CD), crown area (CA)
    pixel_size = 0.1 # (meter)
    CDS_N = rgb_crop.shape[1] * pixel_size
    CDE_W = rgb_crop.shape[2] * pixel_size
    CD = (CDS_N + CDE_W) / 2.0
    CA = CDS_N + CDE_W
    
    # CHM features: H, Hmean, Hstd, PH10, PH25, PH50, PH75, PH90, PH95
    H = np.max(chm_crop)
    Hmean = np.mean(chm_crop)
    Hstd = np.std(chm_crop)
    PH10 = np.percentile(chm_crop, 10)
    PH25 = np.percentile(chm_crop, 25)
    PH50 = np.percentile(chm_crop, 50)
    PH75 = np.percentile(chm_crop, 75)
    PH90 = np.percentile(chm_crop, 90)
    PH95 = np.percentile(chm_crop, 95)     

    # HSI features: EVI, SAVI, MNDVI, GI, MEAN_RED_EDGE, VOG, MRESRI, Datt, PPR, PSRI, SIPI, PRI, ACI, SL
    # hsi_crop = cv2.resize(hsi_crop, (rgb_crop.shape[1], rgb_crop.shape[2]), interpolation=cv2.INTER_CUBIC)

    # --- Helper functions ---
    def safe_div(n, d):
        return n / d if np.abs(d) > 1e-6 else 0.0

    def safe_mean(x):
        return float(np.nanmean(x)) if len(x) > 0 else 0.0
        # return np.sum(x)

    # --- Prepare lists to collect per-pixel values ---
    evi_list, savi_list, mndvi_list, gi_list, vog_list = [], [], [], [], []
    mresri_list, datt_list, ppr_list, psri_list, sipi_list, pri_list, aci_list, sl_list = [], [], [], [], [], [], [], []
    mean_red_edge_list = []  # For Mean Red-edge Reflectance (690–740 nm)

    # Find indices for 690–740 nm
    red_edge_start = np.searchsorted(wavelengths, 0.690)
    red_edge_end = np.searchsorted(wavelengths, 0.740) + 1  # +1 to include 740 nm

    # --- Loop through pixels inside the crown mask ---
    for i in range(hsi_crop.shape[0]):
        for j in range(hsi_crop.shape[1]):
            spectrum = hsi_crop[i, j, :]

            def get_refl(wavelength):
                idx = np.searchsorted(wavelengths, wavelength)
                if idx == 0:
                    return spectrum[0]
                if idx == len(wavelengths):
                    return spectrum[-1]
                lower_wl = wavelengths[idx - 1]
                upper_wl = wavelengths[idx]
                lower_refl = spectrum[idx - 1]
                upper_refl = spectrum[idx]
                if upper_wl == lower_wl:
                    return lower_refl
                fraction = (wavelength - lower_wl) / (upper_wl - lower_wl)
                return lower_refl + fraction * (upper_refl - lower_refl)

            # Interpolated reflectance values
            p_798 = get_refl(0.798)
            p_679 = get_refl(0.679)
            p_482 = get_refl(0.482)
            p_550 = get_refl(0.550)
            p_553 = get_refl(0.553)
            p_750 = get_refl(0.750)
            p_705 = get_refl(0.705)
            p_445 = get_refl(0.445)
            p_740 = get_refl(0.740)
            p_720 = get_refl(0.720)
            p_850 = get_refl(0.850)
            p_710 = get_refl(0.710)
            p_680 = get_refl(0.680)
            p_450 = get_refl(0.450)
            p_695 = get_refl(0.695)
            p_760 = get_refl(0.760)
            p_800 = get_refl(0.800)
            p_570 = get_refl(0.570)
            p_530 = get_refl(0.530)
            p_650 = get_refl(0.650)
            p_690 = get_refl(0.690)

            # Compute vegetation indices safely
            EVI = safe_div(2.5 * (p_798 - p_679), 1 + p_798 + 6 * p_679 - 7.5 * p_482)
            SAVI = safe_div(1.5 * (p_798 - p_679), p_798 + p_679 + 0.5)
            MNDVI = safe_div(p_750 - p_705, p_750 + p_705)
            GI = safe_div(p_798, p_553) - 1 if p_553 > 1e-6 else np.nan
            VOG = safe_div(p_740, p_720)
            MRESRI = safe_div(p_750 - p_445, p_750 + p_445)
            Datt = safe_div(p_850 - p_710, p_850 - p_680)
            PPR = safe_div(p_550 - p_450, p_550 + p_450)
            PSRI = safe_div(p_695, p_760)
            SIPI = safe_div(p_800 - p_450, p_800 + p_680)
            PRI = safe_div(p_570 - p_530, p_570 + p_530)
            ACI = safe_div(p_650, p_550)
            SL = safe_div(p_740 - p_690, 50.0)

            # Mean Red-edge Reflectance for this pixel
            mean_red_edge = np.mean(spectrum[red_edge_start:red_edge_end])
            if not np.isnan(mean_red_edge):
                mean_red_edge_list.append(mean_red_edge)

            # Append if valid
            if not np.isnan(EVI): evi_list.append(EVI)
            if not np.isnan(SAVI): savi_list.append(SAVI)
            if not np.isnan(MNDVI): mndvi_list.append(MNDVI)
            if not np.isnan(GI): gi_list.append(GI)
            if not np.isnan(VOG): vog_list.append(VOG)
            if not np.isnan(MRESRI): mresri_list.append(MRESRI)
            if not np.isnan(Datt): datt_list.append(Datt)
            if not np.isnan(PPR): ppr_list.append(PPR)
            if not np.isnan(PSRI): psri_list.append(PSRI)
            if not np.isnan(SIPI): sipi_list.append(SIPI)
            if not np.isnan(PRI): pri_list.append(PRI)
            if not np.isnan(ACI): aci_list.append(ACI)
            if not np.isnan(SL): sl_list.append(SL)

    # --- Final mean of vegetation indices ---
    EVI = safe_mean(evi_list)
    SAVI = safe_mean(savi_list)
    MNDVI = safe_mean(mndvi_list)
    GI = safe_mean(gi_list)
    VOG = safe_mean(vog_list)
    MRESRI = safe_mean(mresri_list)
    Datt = safe_mean(datt_list)
    PPR = safe_mean(ppr_list)
    PSRI = safe_mean(psri_list)
    SIPI = safe_mean(sipi_list)
    PRI = safe_mean(pri_list)
    ACI = safe_mean(aci_list)
    SL = safe_mean(sl_list)
    Mean690_740 = safe_mean(mean_red_edge_list)

    # PCA on hyperspectral region
    # hsi_flat = hsi_crop.reshape(-1, hsi_crop.shape[2])
    # pca = PCA(n_components=3)
    # pca_components = pca.fit_transform(hsi_flat)
    # pca_features = np.mean(np.abs(pca_components), axis=0)
    
    # Append features for this tree
    features = {
        'H': H, 'Hmean': Hmean, 
        'Hstd': Hstd,
        'PH10': PH10, 'PH25': PH25, 'PH50': PH50, 
        'PH75': PH75, 'PH90': PH90, 'PH95': PH95,
        'CD': CD, 
        'CA': CA,
        'EVI': EVI, 'SAVI': SAVI, 'GI': GI,
        'MNDVI': MNDVI, 'VOG': VOG,
        'MRESRI': MRESRI, 'Datt': Datt, 'PPR': PPR,
        'PSRI': PSRI, 'SIPI': SIPI, 'PRI': PRI,
        'ACI': ACI, 'SL': SL, 'p_550': p_550, 'p_750': p_750,
        'Mean690-740': Mean690_740,
        # 'PCA1': pca_features[0], 'PCA2': pca_features[1], 'PCA3': pca_features[2]
    } 

    return features

def visualize_detections(rgb_path, merged_boxes, carbon_stocks, total_carbon, site, fig_path=None):
    # Read RGB image (shape: (3, H, W)), convert to (H, W, 3)
    with rasterio.open(rgb_path) as src:
        rgb = src.read([1, 2, 3])
        transform = src.transform
    rgb = np.transpose(rgb, (1, 2, 0)).astype(np.uint8).copy()

    left, bottom = transform * (0, rgb.shape[0])
    right, top = transform * (rgb.shape[1], 0)
    extent = [left, right, bottom, top]

    # Draw boxes and annotate with carbon stock
    # for box, carbon in zip(merged_boxes, carbon_stocks):
    #     x1, y1, x2, y2 = map(int, box)
    #     cv2.rectangle(rgb, (x1, y1), (x2, y2), (0, 255, 0), 3)
    #     cv2.putText(rgb, f"{carbon:.1f}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

    # # Show the image with detections and total carbon
    # plt.figure(figsize=(12, 12))
    # plt.imshow(rgb)
    # plt.title(f"Detections (Total Carbon: {total_carbon:.2f})")
    # plt.axis('off')

    # if fig_path is not None:
    #     plt.savefig(fig_path)

    # plt.show()

    # Vẽ ảnh với tọa độ thực
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(rgb, extent=extent)

    # Vẽ bounding boxes
    for box, carbon in zip(merged_boxes, carbon_stocks):
        x1, y1, x2, y2 = map(int, box)
        # Tọa độ địa lý
        geo_x1, geo_y1 = transform * (x1, y1)
        geo_x2, geo_y2 = transform * (x2, y2)

        rect = plt.Rectangle((geo_x1, geo_y1), geo_x2 - geo_x1, geo_y2 - geo_y1,
                            edgecolor='orange', facecolor='none', linewidth=1)
        ax.add_patch(rect)
        # ax.text(geo_x1, geo_y1, f"{carbon:.1f}", color='blue', fontsize=6)

    # ax.set_title(f"Khu vực rừng {site} (Chỉ số hấp thụ carbon: {total_carbon:.2f} kg)")
    ax.set_title(f"{site} (Carbon Stock: {total_carbon:.2f} kg)")
    # ax.set_xlabel("Longitude or Easting")
    # ax.set_ylabel("Latitude or Northing")
    
    plt.axis('off')
    plt.tight_layout()

    if fig_path is not None:
        plt.savefig(fig_path)

    plt.show()

def visualize_with_latlon(image_path, total_carbon, site, boxes=None, save_fig=None):
    """
    Hiển thị ảnh với trục tọa độ theo kinh độ và vĩ độ (EPSG:4326),
    kèm theo các hộp giới hạn nếu có.
    """
    with rasterio.open(image_path) as src:
        img = src.read([1, 2, 3]) if src.count >= 3 else src.read([1])  # RGB hoặc 1 band
        img = np.moveaxis(img, 0, -1)
        transform = src.transform
        crs = src.crs

        height, width = src.height, src.width

        # Tạo lưới tọa độ pixel
        xs = np.arange(width)
        ys = np.arange(height)
        x_coords, y_coords = np.meshgrid(xs, ys)

        # Chuyển từ pixel sang toạ độ gốc (projected CRS)
        easting, northing = rasterio.transform.xy(transform, y_coords, x_coords, offset='center')

        # Chuyển sang EPSG:4326 (WGS84)
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        lon_grid, lat_grid = transformer.transform(easting, northing)

        # Dùng giới hạn tọa độ thật để tạo axis đúng lat/lon
        lon_min, lon_max = lon_grid.min(), lon_grid.max()
        lat_min, lat_max = lat_grid.min(), lat_grid.max()

        # Vẽ ảnh
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(img, extent=(lon_min, lon_max, lat_min, lat_max))
        ax.set_xlabel("Kinh độ")
        ax.set_ylabel("Vĩ độ")
        ax.set_title(f"Khu vực rừng {site} | Chỉ số hấp thụ carbon: {total_carbon} kg")

        # Vẽ hộp giới hạn tán cây nếu có
        if boxes is not None:
            for box in boxes:
                x_min, y_min, x_max, y_max = box
                # Chuyển pixel sang lon/lat
                lon_min_box, lat_min_box = transformer.transform(*rasterio.transform.xy(transform, y_min, x_min))
                lon_max_box, lat_max_box = transformer.transform(*rasterio.transform.xy(transform, y_max, x_max))
                width = lon_max_box - lon_min_box
                height = lat_max_box - lat_min_box
                rect = plt.Rectangle((lon_min_box, lat_min_box), width, height,
                                     linewidth=1.2, edgecolor='lime', facecolor='none')
                ax.add_patch(rect)

        # Hiển thị diện tích
        area_ha = (lon_max - lon_min) * (lat_max - lat_min) * 111 * 111  # xấp xỉ theo đơn vị độ
        print(f"Ảnh chụp khu vực rộng khoảng: {area_ha:.2f} ha")

        plt.grid(True)
        plt.tight_layout()
        if save_fig is not None:
            plt.savefig(save_fig, dpi=300)
        plt.show()

def plot_carbon_distribution(carbon_stocks, fig_path=None):
    plt.figure(figsize=(8, 4))
    plt.hist(carbon_stocks, bins=20, color='skyblue', edgecolor='black')
    plt.xlabel('Individual Tree Carbon Stock')
    plt.ylabel('Count')
    plt.title('Distribution of Carbon Stock per Tree')

    if fig_path is not None:
        plt.savefig(fig_path)

    plt.show()