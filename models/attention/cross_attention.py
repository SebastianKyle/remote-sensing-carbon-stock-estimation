# Cross-Attention Module
# Author: Doan Manh Tan
# Date: 2025-13-02

from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class CrossAttention(nn.Module):
    def __init__(self, in_channels): 
        super(CrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        # self.gamma = nn.Parameter(torch.zeros(1))

        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)
 
    def forward(self, rgb_features, chm_features):
        batch_size, C, H, W = rgb_features.size()
        
        # Compute attention weights
        query = self.query_conv(chm_features).view(batch_size, -1, H * W)
        key = self.key_conv(rgb_features).view(batch_size, -1, H * W)
        attention_weights = self.softmax(torch.bmm(query.permute(0, 2, 1), key))
        attention_weights = F.softmax(attention_weights, dim=-1)
        
        # Apply attention to the value
        value = self.value_conv(rgb_features).view(batch_size, -1, H * W)
        out = torch.bmm(value, attention_weights.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W)

        # out = self.gamma * out + rgb_features
        
        return out
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class CrossAttention(nn.Module):
    def __init__(self, in_channels): 
        super(CrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1) * 0.1)

        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)
 
    def forward(self, rgb_features, chm_features):

        Q = self.query_conv(rgb_features)
        K = self.key_conv(chm_features)
        V = self.value_conv(chm_features)

        attention_map = torch.softmax(torch.einsum("bchw,bcxy->bhwxy", Q, K), dim=-1)
        chm_attended = torch.einsum("bhwxy,bcxy->bchw", attention_map, V)

        fused = rgb_features + self.gamma * chm_attended
        return fused

        # batch_size, C, H, W = rgb_features.size()
        
        # # Compute attention weights
        # query = self.query_conv(chm_features).view(batch_size, -1, H * W)
        # key = self.key_conv(rgb_features).view(batch_size, -1, H * W)
        # attention_weights = self.softmax(torch.bmm(query.permute(0, 2, 1), key))
        # attention_weights = F.softmax(attention_weights, dim=-1)
        
        # # Apply attention to the value
        # value = self.value_conv(chm_features).view(batch_size, -1, H * W)
        # out = torch.bmm(value, attention_weights.permute(0, 2, 1))
        # out = out.view(batch_size, C, H, W)

        # # out = self.gamma * out + rgb_features
        
        # return out

class CrossAttentionConcat(nn.Module):
    def __init__(self, in_channels): 
        super(CrossAttentionConcat, self).__init__()
        self.query_conv = nn.Conv2d(in_channels * 2, in_channels // 4, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels * 2, in_channels // 4, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

        self.softmax = nn.Softmax(dim=-1)
        # self.gamma = nn.Parameter(torch.ones(1) * 0.1)

        self.channel_reduction = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)
 
    def forward(self, rgb_features, chm_features):
        combined_features = torch.cat([rgb_features, chm_features], dim=1)

        # Compute Query, Key, and Value from combined features
        Q = self.query_conv(combined_features)
        K = self.key_conv(combined_features)
        V = self.value_conv(combined_features)

        # Compute attention scores
        attention_map = self.softmax(torch.einsum("bchw,bcxy->bhwxy", Q, K))

        # Apply attention to the value tensor
        attended_features = torch.einsum("bhwxy,bcxy->bchw", attention_map, V)

        fused = self.channel_reduction(attended_features)

        return fused