import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseModel(nn.Module, ABC):
    def __init__(self):
        super(BaseModel, self).__init__()
        
    @abstractmethod
    def forward(self, x):
        pass
    
    @abstractmethod
    def train_model(self, train_loader, optimizer, criterion, device):
        pass

    @abstractmethod
    def evaluate_model(self, test_loader, device):
        pass