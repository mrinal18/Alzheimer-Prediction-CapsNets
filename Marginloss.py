#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 17 18:23:01 2018

@author: science
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F



class MarginLoss(nn.Module):
    def __init__(self):
        # Reconstruction as regularization
        self.reconstruction_loss = nn.MSELoss(size_average=False)

    def forward(self, images, labels, classes, reconstructions):
        left = F.relu(0.9 - classes, inplace=True) ** 2
        right = F.relu(classes - 0.1, inplace=True) ** 2
        margin_loss = labels * left + 0.5 * (1. - labels) * right
        margin_loss = margin_loss.sum()
        reconstruction_loss = self.reconstruction_loss(reconstructions, images)

        return (margin_loss + 0.0005 * reconstruction_loss) / images.size(0)