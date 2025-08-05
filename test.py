import os
import torch
import yaml
import models
from models.DeepForest.wrapper import DeepForestWrapper
import datasets as ds
from train import collate_fn

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    if device.type == "cuda":
        models.get_gpu_prop(show=True)
    print("\ndevice: {}".format(device))
    
    # ------------------------ Data Loader ------------------------ #
    data_dir = args.data_dir
    dataset_test = ds.TreeDataset(data_dir, args.max_workers, args.verbose, train=False)
    d_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.max_workers, collate_fn=collate_fn)
     
    # ------------------------------------------------------------- #
    print(args)
    num_classes = max(d_test.dataset.classes) + 1

    anchor_sizes = getattr(args, 'anchor_sizes', [32, 64, 128, 256, 512])  
    anchor_sizes = tuple((s,) for s in anchor_sizes)

    # Initialize model based on model type
    if args.model.lower() == 'deepforest_retinanet':
        print("\nEvaluating DeepForest RetinaNet model...")
        model = DeepForestWrapper(model_type='retinanet').to(device)
    elif args.model == 'faster_rcnn':
        model = models.fasterrcnn_fpn_resnet50(pretrained=False, num_classes=num_classes).to(device)
    elif args.model == 'hcf_rcnn':
        model = models.fused_rcnn_resnet50(pretrained=True, num_classes=num_classes, backbone_name='hcf_resnet50', anchor_sizes=anchor_sizes).to(device)
    elif args.model == 'yolov12':
        model = models.YOLOv12(
            num_classes=num_classes-1, 
            depth_multiple=0.5, width_multiple=1.00, 
            max_channels=512, 
            pretrained=True, path='yolo12m.pt'
        ).to(device)

    # Load checkpoint only for our custom models, not for DeepForest models
    if not args.model.lower().startswith('deepforest'):
        print(f"\nLoading checkpoint from {args.ckpt_path}")
        checkpoint = torch.load(args.ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model"])

    # Set model to evaluation mode
    model.eval()

    # Evaluate the model
    print("\nRunning evaluation...")
    eval_output = models.evaluate_model(model, d_test, device, args=args)
    
    # Print evaluation results
    print("\nEvaluation Results:")
    print(f"Precision: {eval_output['precision']:.4f}")
    print(f"Recall: {eval_output['recall']:.4f}")
    print(f"F1 score: {eval_output['f1_score']:.4f}")
    print(f"AP: {eval_output['average_precision']:.4f}") 
    print(f"Mean IoU: {eval_output['mean_iou']:.4f}")
    # print(f"Mean CIoU: {eval_output['mean_ciou']:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yml", help="path to the config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    for key, value in config['test'].items():
        setattr(args, key, value)
    
    if args.ckpt_path is None and not args.model.lower().startswith('deepforest'):
        args.ckpt_path = "./fasterrcnn_{}.pth".format(args.dataset)
    
    main(args)