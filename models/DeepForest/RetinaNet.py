import torch
import torchvision
from torchvision.models.detection.retinanet import RetinaNet, AnchorGenerator, RetinaNet_ResNet50_FPN_Weights
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

def deep_forest_retinanet():
    backbone = torchvision.models.detection.retinanet_resnet50_fpn(
        weights=RetinaNet_ResNet50_FPN_Weights.COCO_V1
    )
    backbone = backbone.backbone

    model = RetinaNet(
        backbone=backbone,
        num_classes=1,  # background + tree
    )

    model.score_thresh = 0.1  # From their config for tree detection
    model.nms_thresh = 0.2  # From their config

    try:
        # First try to load safetensors file
        weights_path = hf_hub_download(
            repo_id="weecology/deepforest-tree",
            filename="model.safetensors"
        )
        state_dict = load_file(weights_path)
        
        # Remove the "model." prefix from keys
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('model.'):
                new_key = key[6:]  # Remove 'model.' prefix
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value

        # Load the weights
        model.load_state_dict(new_state_dict, strict=True)
        print("Successfully loaded weights from safetensors file")
        
    except Exception as e:
        print(f"Error loading safetensors: {e}")
        try:
            # If that fails, try the .pt file
            weights_path = hf_hub_download(
                repo_id="weecology/deepforest-tree",
                filename="NEON.pt"
            )
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)  # Set weights_only=True
            model.load_state_dict(state_dict, strict=True)
            print("Successfully loaded weights from NEON.pt file")
        except Exception as e:
            print(f"Error loading weights: {e}")
            print("Using default pretrained weights")

    return model