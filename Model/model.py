# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 21:29:42 2026

@author: quent

This class is a very basic 2 layer 2 dimensions Convolutional Network
"""

import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)  # 28 -> 14
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)  # 14 -> 7
        x = x.flatten(1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)
