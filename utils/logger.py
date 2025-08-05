import matplotlib.pyplot as plt
import json
import os
import numpy as np
import torch

class Logger:
    def __init__(self, log_dir, model_name):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.model_name = model_name
        self.train_losses = []
        self.precisions = []
        self.recalls = []
        self.mean_ious = []
        self.f1_scores = []
        self.average_precisions = []

    def log_train_loss(self, loss):
        self.train_losses.append(loss)

    def log_metrics(self, precision, recall, mean_iou, f1_score, average_precision):
        self.precisions.append(convert_to_serializable(precision))
        self.recalls.append(convert_to_serializable(recall))
        self.mean_ious.append(convert_to_serializable(mean_iou))
        self.f1_scores.append(convert_to_serializable(f1_score))
        self.average_precisions.append(convert_to_serializable(average_precision))

    def save_logs(self, filename):
        logs = {
            "model_name": self.model_name,
            "train_losses": self.train_losses,
            "precisions": self.precisions,
            "recalls": self.recalls,
            "mean_ious": self.mean_ious,
            "f1_scores": self.f1_scores,
            "average_precisions": self.average_precisions
        }
        with open(os.path.join(self.log_dir, f"{filename}.json"), "w") as f:
            json.dump(logs, f)

    def load_logs(self, log_file):
        with open(log_file, "r") as f:
            logs = json.load(f)
        self.model_name = logs["model_name"]
        self.train_losses = logs["train_losses"]
        self.precisions = logs["precisions"]
        self.recalls = logs["recalls"]
        self.mean_ious = logs["mean_ious"]
        self.f1_scores = logs["f1_scores"]
        self.average_precisions = logs["average_precisions"]

    @staticmethod
    def plot_metric(log_files, metric, save_path=None, labels=None):
        plt.figure(figsize=(12, 8))

        for i, log_file in enumerate(log_files):
            with open(log_file, "r") as f:
                logs = json.load(f)
            epochs = range(1, len(logs["train_losses"]) + 1)

            plt.plot(epochs, logs[metric], label=f'{logs["model_name"]}')

        plt.xlabel('Epochs')
        plt.ylabel(metric.replace("_", " ").title())
        plt.legend()
        plt.title(f"{metric.replace('_', ' ').title()} Comparison")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        plt.show()

def convert_to_serializable(obj):
        if isinstance(obj, np.float32):  # Check for NumPy float32
            return float(obj)
        elif isinstance(obj, torch.Tensor):  # Check for PyTorch tensors
            return obj.item() if obj.numel() == 1 else obj.tolist()
        elif isinstance(obj, dict):  # Recursively handle dictionaries
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):  # Recursively handle lists
            return [convert_to_serializable(v) for v in obj]
        return obj  # Return the object as-is if no conversion is needed