#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 17 18:24:23 2018

@author: science
"""

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from CapsuleLayer import CapsuleLayer


class CapsuleNet(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        self.conv1 = nn.Conv3d(3, 64, kernel_size=11, stride=4, padding=21)
        self.primary_capsules = CapsuleLayer(
            8, -1, 256, 32, kernel_size=9, stride=2)

        # 10 is the number of classes
        self.digit_capsules = CapsuleLayer(10, 32 * 6 * 6, 8, 64)

        self.decoder = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
            nn.Sigmoid(),
        )

    def forward(self, x, y=None):
        x = F.relu(self.conv1(x), inplace=True)
        x = self.primary_capsules(x)
        x = self.digit_capsules(x).squeeze().transpose(0, 1)

        classes = (x ** 2).sum(dim=-1) ** 0.5
        classes = F.softmax(classes)

        if y is None:
            # In all batches, get the most active capsule
            _, max_length_indices = classes.max(dim=1)
            y = Variable(torch.eye(10)).cuda().index_select(
                dim=0, index=max_length_indices.data)

        reconstructions = self.decoder((x * y[:, :, None]).view(x.size(0), -1))
        return classes, reconstructions