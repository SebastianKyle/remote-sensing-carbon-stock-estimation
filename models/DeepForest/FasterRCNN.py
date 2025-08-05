import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FasterRCNN, FastRCNNPredictor
from huggingface_hub import hf_hub_download

def deep_forest_faster_rcnn():
    # 1. Create the model architecture
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights='FasterRCNN_ResNet50_FPN_Weights.COCO_V1'
    )

    # Modify the box predictor for your number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, 
        num_classes=2  # 1 class + background
    )

    # 2. Download weights from Hugging Face (same as above)
    weights_path = hf_hub_download(
        repo_id="weecology/deepforest-tree",  # or other model
        filename="pytorch_model.bin"
    )

    # 3. Load the weights
    state_dict = torch.load(weights_path)
    model.load_state_dict(state_dict)

    return model