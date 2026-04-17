import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class CenterNetHead(nn.Module):
    def __init__(self, in_channels, num_classes=1):
        super(CenterNetHead, self).__init__()
        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )
        self.wh_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 2, kernel_size=1)
        )
        self.reg_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 2, kernel_size=1)
        )

        # --- NEW CODE: Initialize prior probability ---
        # This forces the model to start with a ~0.01 prediction for objects,
        # preventing background loss explosion.
        self.cls_head[-1].bias.data.fill_(-2.19)

    def forward(self, x):
        # x is the fused feature from MICPL at the final time step T
        hm = torch.sigmoid(self.cls_head(x)) # Clamp to 0-1 for probabilities
        wh = self.wh_head(x)
        reg = self.reg_head(x)
        return {'hm': hm, 'wh': wh, 'reg': reg}

def focal_loss(pred, gt):
    """Modified focal loss for CenterNet heatmap."""
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()
    
    neg_weights = torch.pow(1 - gt, 6)
    
    loss = 0
    pred = torch.clamp(pred, 1e-4, 1 - 1e-4) # Prevent log(0)
    
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds
    
    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    
    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss
