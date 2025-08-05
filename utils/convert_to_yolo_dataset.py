import os
import json
import cv2
from sklearn.model_selection import train_test_split
from datasets.dataset import TreeDataset

def convert_to_yolo_format(dataset, output_dir):
    """
    Convert dataset to YOLO format (images and labels).
    """
    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    for idx in range(len(dataset)):
        rgb_image, _, target = dataset[idx]

        # Load image and its size
        img_id = dataset.ids[idx]
        img_path = dataset.imgs[img_id]['rgb_path']
        img = cv2.imread(img_path)
        height, width, _ = img.shape

        # Save image
        new_img_path = os.path.join(images_dir, f"{img_id}.jpg")
        cv2.imwrite(new_img_path, img)

        # Convert bounding boxes
        yolo_labels = []
        for box in target["boxes"]:
            xmin, ymin, xmax, ymax = box.tolist()
            x_center = (xmin + xmax) / (2 * width)
            y_center = (ymin + ymax) / (2 * height)
            w = (xmax - xmin) / width
            h = (ymax - ymin) / height
            yolo_labels.append(f"0 {x_center} {y_center} {w} {h}")

        # Save label file
        label_path = os.path.join(labels_dir, f"{img_id}.txt")
        with open(label_path, "w") as f:
            f.write("\n".join(yolo_labels))

    print(f"YOLO dataset saved in {output_dir}")

def convert_to_yolo_format_with_split(dataset, output_dir, subset_ids):
    """
    Convert a dataset subset (train or val) to YOLO format.
    """
    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    for img_id in subset_ids:
        rgb_image, _, target = dataset[dataset.ids.index(img_id)]

        # Load image and its size
        img_path = dataset.imgs[img_id]['rgb_path']
        img = cv2.imread(img_path)
        height, width, _ = img.shape

        # Save image
        new_img_path = os.path.join(images_dir, f"{img_id}.jpg")
        cv2.imwrite(new_img_path, img)

        # Convert bounding boxes to YOLO format
        yolo_labels = []
        for box in target["boxes"]:
            xmin, ymin, xmax, ymax = box.tolist()
            x_center = (xmin + xmax) / (2 * width)
            y_center = (ymin + ymax) / (2 * height)
            w = (xmax - xmin) / width
            h = (ymax - ymin) / height
            yolo_labels.append(f"0 {x_center} {y_center} {w} {h}")

        # Save label file
        label_path = os.path.join(labels_dir, f"{img_id}.txt")
        with open(label_path, "w") as f:
            f.write("\n".join(yolo_labels))

    print(f"YOLO dataset saved in {output_dir}")

# Load the full training dataset
dataset = TreeDataset("/media02/lqngoc22/thesis-tree-delineation/data/preprocessed/", train=True)

# Split dataset into 90% train, 10% validation
train_ids, val_ids = train_test_split(dataset.ids, test_size=0.1, random_state=42)

# Convert train and validation sets to YOLO format
convert_to_yolo_format_with_split(dataset, "/media02/lqngoc22/thesis-tree-delineation/data/yolo/train/", train_ids)
convert_to_yolo_format_with_split(dataset, "/media02/lqngoc22/thesis-tree-delineation/data/yolo/val/", val_ids)

test_dataset = TreeDataset("/media02/lqngoc22/thesis-tree-delineation/data/preprocessed/", train=False)
convert_to_yolo_format(test_dataset, "/media02/lqngoc22/thesis-tree-delineation/data/yolo/test/")