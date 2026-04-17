# ============================================================
# CELL: DLA-34 Backbone
# Replaces SimpleFeatureExtractor in the original notebook.
# DLA-34 was the backbone used in the original CenterNet paper.
# Insert this AFTER the MICPL architecture cell (cell 1).
# ============================================================

import torch
import torch.nn as nn
import timm
from torch.hub import load_state_dict_from_url

# ------------------------------------------------------------------
# DLA-34 Feature Extractor
# Uses timm's pretrained DLA-34. The network naturally outputs
# multi-scale features; we take the final output at stride 4
# (same resolution as the original SimpleFeatureExtractor: 128x128
# for 512 input) and project to `out_channels` to keep MICPL
# input dimensions identical to before.
# ------------------------------------------------------------------
class DLA34FeatureExtractor(nn.Module):
    def __init__(self, out_channels=64, pretrained=True, freeze_backbone=True):
        super().__init__()
        DLA34_URL = "http://dl.yf.io/dla/models/imagenet/dla34-ba72cf86.pth"

        # self.dla = timm.create_model(
        #     'dla34',
        #     pretrained=False,
        #     features_only=True,
        #     out_indices=(1, 2, 3),
        # )

        # if pretrained:
        #     state_dict = torch.hub.load_state_dict_from_url(
        #         DLA34_URL, map_location='cpu', check_hash=False
        #     )
        #     self.dla.load_state_dict(state_dict, strict=False)
        #     print("DLA-34 weights loaded successfully.")
        self.dla=timm.create_model("hf_hub:timm/dla34.in1k", pretrained=pretrained,features_only=True,out_indices=(1,2,3))

        if freeze_backbone:
            self.dla.requires_grad_(False)
            print("DLA-34 backbone frozen.")

        channels = self.dla.feature_info.channels()
        c1, c2, c3 = channels

        self.up_c3 = nn.Conv2d(c3, out_channels, kernel_size=1, bias=False)
        self.up_c2 = nn.Conv2d(c2, out_channels, kernel_size=1, bias=False)
        self.up_c1 = nn.Conv2d(c1, out_channels, kernel_size=1, bias=False)

        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def unfreeze_backbone(self):
        self.dla.requires_grad_(True)
        print("Backbone unfrozen.")
    def forward(self, x):
        with torch.no_grad():
            f1, f2, f3 = self.dla(x)   # no gradients computed through backbone

        target_size = f2.shape[-2:]

        p3 = F.interpolate(self.up_c3(f3), size=target_size, mode='bilinear', align_corners=False)
        p2 = self.up_c2(f2)
        p1 = F.avg_pool2d(self.up_c1(f1), 2)

        return self.fuse(p1 + p2 + p3)


# ------------------------------------------------------------------
# Updated SmallObjectDetector using DLA-34
# Drop-in replacement — output shape identical to original.
# ------------------------------------------------------------------
#class SmallObjectDetector(nn.Module):
#    def __init__(self, backbone, channels=64, T=5):
#        super().__init__()
#        self.backbone = backbone
#        self.micpl = MICPL_Module(channels, T)
#
#    def forward(self, x_sequence, training=True):
#        # x_sequence: [B, 3, T, H, W]
#        vision_features = []
#        for t in range(x_sequence.size(2)):
#            feat = self.backbone(x_sequence[:, :, t, :, :])
#            vision_features.append(feat)
#
#        x_s = torch.stack(vision_features, dim=2)  # [B, C, T, H/4, W/4]
#
#        if training:
#            h_s = self.micpl(x_s)
#            return x_s + h_s
#        else:
#            return x_s
