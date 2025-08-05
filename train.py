import bisect
import glob
import os
import re
import time
import yaml

import torch

import models
import datasets as ds
from utils.logger import Logger

from config import load_config

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    rgb_images, chm_images, targets = zip(*batch)
    return list(rgb_images), list(chm_images), list(targets)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    if device.type == "cuda":
        models.get_gpu_prop(show=True)
    print("\ndevice: {}".format(device))

    # ------------------------ Data Loader ------------------------ #
    data_dir = args.data_dir
    dataset = ds.TreeDataset(data_dir, args.max_workers, args.verbose)

    d_train = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.max_workers, collate_fn=collate_fn) 

    dataset_test = ds.TreeDataset(data_dir, args.max_workers, args.verbose, train=False)
    d_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.max_workers, collate_fn=collate_fn)

    args.warmup_iters = max(1000, len(d_train))

    # ------------------------------------------------------------- #
    print(args)
    num_classes = max(dataset.classes) + 1
    detr_criterion = None
    anchor_sizes = getattr(args, 'anchor_sizes', [32, 64, 128, 256, 512])  
    anchor_sizes = tuple((s,) for s in anchor_sizes)
    if args.model == 'faster_rcnn':
        model = models.fasterrcnn_fpn_resnet50(pretrained=True, num_classes=num_classes, args=args).to(device)
    elif args.model == 'hcf_rcnn':
        model = models.fused_rcnn_resnet50(pretrained=True, num_classes=num_classes, backbone_name='hcf_resnet50', anchor_sizes=anchor_sizes, loss_cfg=args.loss_weights).to(device)
    elif args.model == 'yolov12':
        model = models.YOLOv12(num_classes=num_classes-1, width_multiple=1.00, depth_multiple=0.5, max_channels=512, pretrained=True, path='yolo12m.pt').to(device)
    elif args.model == 'detr':
        # Initialize DETR with specified backbone
        backbone = args.backbone if hasattr(args, 'backbone') else 'resnet50'
        num_queries = args.num_queries if hasattr(args, 'num_queries') else 100
        model = models.build_detr(
            num_classes=num_classes-1,  # DETR doesn't need background class
            backbone=backbone,
            num_queries=num_queries,
            pretrained=True
        ).to(device)

        detr_criterion = models.DETRLoss(
            num_classes=num_classes-1, 
            matcher=models.HungarianMatcher(), 
            weight_dict={'loss_ce': 1, 'loss_bbox': 5, 'loss_giou': 2}, 
            eos_coef=0.1, 
            losses=['labels', 'boxes', 'cardinality']
        )

    # Optimizer setup with different parameters for backbone and rest of model
    if args.model == 'detr':
        param_dicts = [
            {
                "params": [p for n, p in model.named_parameters() 
                          if "backbone" not in n and p.requires_grad],
                "lr": args.lr,
            },
            {
                "params": [p for n, p in model.named_parameters() 
                          if "backbone" in n and p.requires_grad],
                "lr": args.lr * 0.1,  # Lower LR for backbone
            },
        ]
        params = [p for group in param_dicts for p in group["params"]]  # Flatten params for grad clipping
    else:
        params = [p for p in model.parameters() if p.requires_grad]
    
    # Optimizer setup
    if args.optimizer.lower() == 'sgd':
        optimizer = torch.optim.SGD(
            param_dicts if args.model == 'detr' else params,  # Use param_dicts for DETR
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    elif args.optimizer.lower() in ['adam', 'adamw']:  # Adam optimizer
        optimizer_class = torch.optim.AdamW if args.optimizer.lower() == 'adamw' else torch.optim.Adam
        # Ensure proper type conversion for Adam parameters
        
        optimizer = optimizer_class(
            param_dicts if args.model == 'detr' else params, # Use param_dicts for DETR
            lr=float(args.lr),
            weight_decay=float(args.weight_decay)
        )

    # Learning rate scheduler
    if args.lr_scheduler == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=args.lr_drop, 
            gamma=0.1
        )
    elif args.lr_scheduler == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.lr * 0.01
        )
    elif args.lr_scheduler == 'onecycle':
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr * 10,  # Peak learning rate for one cycle
            epochs=args.epochs,
            steps_per_epoch=len(d_train),
            pct_start=0.3,
            anneal_strategy='cos'
        )

    # Gradient Clipping
    if args.grad_clip > 0 and args.model == 'detr':
        torch.nn.utils.clip_grad_norm_(params, args.grad_clip)

    start_epoch = 0
    
    # find all checkpoints, load the latest checkpoint
    prefix, ext = os.path.splitext(args.ckpt_path)
    ckpts = glob.glob(prefix + "-*" + ext)
    
    # Filter and sort checkpoints
    ckpts = [ckpt for ckpt in ckpts if re.search(r"-(\d+){}".format(ext), os.path.split(ckpt)[1])]
    ckpts.sort(key=lambda x: int(re.search(r"-(\d+){}".format(ext), os.path.split(x)[1]).group(1)) 
               if re.search(r"-(\d+){}".format(ext), os.path.split(x)[1]) else float('inf'))
    
    if ckpts:
        checkpoint = torch.load(ckpts[-1], map_location=device) # load last checkpoint
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epochs"]
        del checkpoint
        torch.cuda.empty_cache()

    since = time.time()
    print("\nalready trained: {} epochs; to {} epochs".format(start_epoch, args.epochs))

    model_name_dict = {
        'faster_rcnn': 'Faster R-CNN',
        'hcf_rcnn': 'HCF R-CNN',
        'detr': 'DETR',
        'yolov12': 'YOLOv12',
    }
    model_name = model_name_dict[args.model]
    logger = Logger(log_dir=args.log_dir, model_name=model_name) 

    best_val_metric = float('-inf')
    best_recall = float('-inf')
    best_precision = float('-inf')

    # ------------------------ Training ------------------------ #
    
    for epoch in range(start_epoch, args.epochs):
        print("\nepoch: {}".format(epoch + 1))
        
        A = time.time()
        # Get current learning rate before training
        current_lr = optimizer.param_groups[0]['lr']
        print("Current learning rate: {:.8f}".format(current_lr))
        
        # Train for one epoch
        iter_train, train_loss = models.train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=d_train,
            device=device,
            epoch=epoch,
            args=args,
            detr_criterion=detr_criterion,
        )
        
        # Step the scheduler after training
        if args.lr_scheduler == 'step':
            lr_scheduler.step()
        elif args.lr_scheduler == 'cosine' or args.lr_scheduler == 'onecycle':
            lr_scheduler.step()
            next_lr = optimizer.param_groups[0]['lr']
            if next_lr != current_lr:
                print(f"Learning rate updated to: {next_lr:.5f}")
        
        A = time.time() - A
        
        B = time.time()
        eval_output = models.evaluate_model(model, d_test, device, args=args)
        B = time.time() - B
        
        trained_epoch = epoch + 1
        print("training: {:.1f} s, evaluation: {:.1f} s".format(A, B))
        models.collect_gpu_info("fasterrcnn", [1 / iter_train, 1 / (B / len(d_test))])

        print(f"Precision: {eval_output['precision']:.4f}")
        print(f"Recall: {eval_output['recall']:.4f}")
        print(f"F1 score: {eval_output['f1_score']:.4f}")
        print(f"AP: {eval_output['average_precision']:.4f}")
        print(f"Mean IoU: {eval_output['mean_iou']:.4f}")

        logger.log_train_loss(train_loss)
        logger.log_metrics(eval_output['precision'], eval_output['recall'], eval_output['mean_iou'], eval_output['f1_score'], eval_output['average_precision'])

        models.save_ckpt(model, optimizer, trained_epoch, args.ckpt_path, eval_info=str(eval_output))

        if eval_output['average_precision'] > best_val_metric:
            best_val_metric = eval_output['average_precision']
            best_ckpt_path = F"{prefix}{ext}"
            models.save_best_ckpt(model, optimizer, trained_epoch, best_ckpt_path, eval_info=str(eval_output))

        if eval_output['recall'] > best_recall:
            best_recall = eval_output['recall']
            best_recall_ckpt_path = F"{prefix}{ext}"
            models.save_best_ckpt(model, optimizer, trained_epoch, best_recall_ckpt_path, eval_info=str(eval_output), best_recall=True)

        if eval_output['precision'] > best_precision:
            best_precision = eval_output['precision']
            best_precision_ckpt_path = F"{prefix}{ext}"
            models.save_best_ckpt(model, optimizer, trained_epoch, best_precision_ckpt_path, eval_info=str(eval_output), best_precision=True)

        # Keep only the last n checkpoints
        prefix, ext = os.path.splitext(args.ckpt_path)
        ckpts = glob.glob(prefix + "-*" + ext)
        ckpts = [ckpt for ckpt in ckpts if re.search(r"-(\d+){}".format(ext), os.path.split(ckpt)[1])]
        ckpts.sort(key=lambda x: int(re.search(r"-(\d+){}".format(ext), os.path.split(x)[1]).group(1)) 
                   if re.search(r"-(\d+){}".format(ext), os.path.split(x)[1]) else float('inf'))
        n = 1
        if len(ckpts) > n:
            for i in range(len(ckpts) - n):
                os.system("rm {}".format(ckpts[i]))

    # Final evaluation on the test dataset (load the best checkpoint)
    best_ckpt_path = F"{prefix}_best{ext}"
    checkpoint = torch.load(best_ckpt_path, map_location=device) 
    model.load_state_dict(checkpoint["model"])
    model.eval()

    final_eval_output = models.evaluate_model(model, d_test, device, args=args)
    print("\nFinal evaluation on the test dataset:")
    print(f"Precision: {final_eval_output['precision']:.4f}")
    print(f"Recall: {final_eval_output['recall']:.4f}")
    print(f"F1 score: {final_eval_output['f1_score']:.4f}")
    print(f"AP: {final_eval_output['average_precision']:.4f}")
    print(f"Mean IoU: {final_eval_output['mean_iou']:.4f}")

    # ------------------------------------------------------------- #
    print("\ntotal time of this training: {:.1f} s".format(time.time() - since))
    if start_epoch < args.epochs:
        print("already trained: {} epochs\n".format(trained_epoch))

    # Save logs with filename of model name and datetime of training attempt
    logger.save_logs(filename=f"{args.model}_{time.strftime('%Y%m%d-%H%M%S')}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yml", help="path to the config file")
    
    # Add all possible training arguments that can be overridden
    parser.add_argument("--dataset", type=str, help="dataset name")
    parser.add_argument("--data_dir", type=str, help="path to data directory")
    parser.add_argument("--model", type=str, help="model name")
    parser.add_argument("--num_classes", type=int, help="number of classes")
    parser.add_argument("--ckpt_path", type=str, help="path to save checkpoints")
    parser.add_argument("--results", type=str, help="path to save results")
    parser.add_argument("--max_workers", type=int, help="maximum number of workers")
    parser.add_argument("--batch_size", type=int, help="batch size")
    parser.add_argument("--seed", type=int, help="random seed")
    parser.add_argument("--lr", type=float, help="learning rate")
    parser.add_argument("--momentum", type=float, help="momentum")
    parser.add_argument("--weight_decay", type=float, help="weight decay")
    parser.add_argument("--epochs", type=int, help="number of epochs")
    parser.add_argument("--lr_steps", type=list, help="learning rate steps")
    parser.add_argument("--lr_drop", type=int, help="learning rate drop")
    parser.add_argument("--iters", type=int, help="number of iterations")
    parser.add_argument("--print_freq", type=int, help="print frequency")
    parser.add_argument("--log_dir", type=str, help="log directory")
    parser.add_argument("--optimizer", type=str, help="optimizer name")
    parser.add_argument("--lr_scheduler", type=str, help="learning rate scheduler")
    parser.add_argument("--grad_clip", type=float, help="gradient clipping")
    parser.add_argument("--warmup_iters", type=int, help="warmup iterations")
    parser.add_argument("--experiment", type=str, help="experiment name for wandb")
    parser.add_argument("--frcnn_input", type=str, help="Input type for faster r-cnn model (rgb/chm/both)")
    
    # Parse command line arguments first
    args = parser.parse_args()
    
    # Load config file
    config = load_config(args.config)

    # Set default values from config file
    for key, value in config['training'].items():
        if not hasattr(args, key) or getattr(args, key) is None:
            setattr(args, key, value)

    # Attach loss_weights to args for easy access
    args.loss_weights = config['training'].get('loss_weights', {})
    
    # Set default values for optional arguments
    if args.lr is None:
        args.lr = 0.02 * 1 / 16  # lr should be 'batch_size / 16 * 0.02'
    if args.ckpt_path is None:
        args.ckpt_path = "./fasterrcnn_{}.pth".format(args.dataset)
    if args.results is None:
        args.results = os.path.join(os.path.dirname(args.ckpt_path), "fasterrcnn_results.pth")
    
    main(args)