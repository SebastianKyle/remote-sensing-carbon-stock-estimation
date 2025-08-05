import torch
from torch import nn

from .RetinaNet import deep_forest_retinanet
from .FasterRCNN import deep_forest_faster_rcnn

class DeepForestWrapper(nn.Module):
    """
    Wrapper class to make DeepForest models compatible with the evaluation pipeline.
    """
    def __init__(self, model_type='retinanet'):
        super().__init__()
        if model_type.lower() == 'retinanet':
            self.model = deep_forest_retinanet()
        elif model_type.lower() == 'fasterrcnn':
            self.model = deep_forest_faster_rcnn()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model.eval()  # Set to evaluation mode by default

    def forward(self, rgb_images, chm_images=None, targets=None):
        """
        Forward pass that matches your custom RCNN interface.
        
        Args:
            rgb_images: RGB images (either single image or list)
            chm_images: CHM images (ignored for DeepForest models)
            targets: Optional target annotations
        """
        # Ensure input is in the correct format
        if not isinstance(rgb_images, (list, tuple)):
            rgb_images = [rgb_images]

        # Convert to the format expected by torchvision models
        if isinstance(rgb_images, list):
            rgb_images = torch.stack(rgb_images)

        # Forward pass through the model
        with torch.no_grad():
            outputs = self.model(rgb_images)

        if self.training and targets is not None:
            # During training, return losses (though these models won't be trained)
            return outputs
        else:
            # During inference, format outputs to match your custom RCNN format
            formatted_outputs = []
            for output in outputs:
                formatted_output = {
                    'boxes': output['boxes'],
                    'labels': output['labels'],
                    'scores': output['scores']
                }
                formatted_outputs.append(formatted_output)
            
            # return formatted_outputs[0] if len(formatted_outputs) == 1 else formatted_outputs
            return formatted_outputs

    def eval(self):
        """Set the model to evaluation mode."""
        self.training = False
        self.model.eval()
        return self

    def train(self, mode=True):
        """Set the model to training mode (though these models won't be trained)."""
        self.training = mode
        self.model.train(mode)
        return self 