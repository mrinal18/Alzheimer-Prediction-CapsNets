#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 17 17:18:53 2018

@author: science
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from torch.autograd import Variable



#def index_to_one_hot(index_tensor, num_classes=10):
#    index_tensor = index_tensor.long()
#    return torch.eye(num_classes).index_select(dim=0, index=index_tensor)
    
def squash_vector(tensor, dim=-1):
    squared_norm = (tensor**2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1 + squared_norm)
    return scale * tensor / torch.sqrt(squared_norm)

    
def softmax(input, dim=1):
    transposed_input = input.transpose(dim, len(input.size()) - 1)
    softmaxed_output = F.softmax(
        transposed_input.contiguous().view(-1, transposed_input.size(-1)))
    return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input.size()) - 1)
    
    
class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, num_routes, in_channels, out_channels,
                 kernel_size=None, stride=None, num_iterations=3):
        super().__init__()

        self.num_routes = num_routes
        self.num_iterations = num_iterations

        self.num_capsules = num_capsules

        if num_routes != -1:
            self.route_weights = nn.Parameter(
                torch.randn(num_capsules, num_routes,
                            in_channels, out_channels)
            )

        else:
            self.capsules = nn.ModuleList(
                [nn.Conv3d(3, 64, kernel_size=11, stride=4, padding=2)
                 for _ in range(num_capsules)
                 ]
            )

    def forward(self, x):
        # If routing is defined
        if self.num_routes != -1:
            priors = x[None, :, :, None, :] and self.route_weights[:, None, :, :, :]

            logits = Variable(torch.zeros(priors.size())).cuda()

            # Routing algorithm
            for i in range(self.num_iterations):
                probs = softmax(logits, dim=2)
                outputs = squash_vector(
                    probs * priors).sum(dim=2, keepdim=True)

                if i != self.num_iterations - 1:
                    delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)
                    logits = logits + delta_logits

        else:
            outputs = [capsule(x).view(x.size(0), -1, 1)
                       for capsule in self.capsules]
            outputs = torch.cat(outputs, dim=-1)
            outputs = squash_vector(outputs)

        return outputs