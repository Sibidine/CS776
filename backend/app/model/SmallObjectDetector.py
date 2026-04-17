
import torch
import torch.nn as nn
import timm
from torch.hub import load_state_dict_from_url

class SmallObjectDetector(nn.Module):
    """
    Architecture:

        raw frames
            ├──► DLA-34 backbone  ──► x_s  [B, C, T, H, W]
            │                                      │
            └──► RAFT (parallel)  ──► flows         │
                   (T-1 pairs)          │           │
                                        ▼           │
                                  warp x_s[:,t]     │
                                  using flow[t-1]   │
                                        │           │
                                        └─── x_s_warped ──► MICPL ──► h_s
                                                                        │
                                                          x_s + h_s ◄──┘
                                                          (fusion)
    """
    def __init__(self, backbone, raft, channels=64, T=5):
        super().__init__()
        self.backbone = backbone
        self.raft     = raft
        self.micpl    = MICPL_Module(channels, T)
        self.T        = T

    def forward(self, x_sequence, training=True):
        """
        Args:
            x_sequence: [B, 3, T, H, W]  — raw RGB frames, values in [0, 1]
        """
        B, _, T, H, W = x_sequence.shape

        # ── Branch 1: DLA-34 backbone (per frame) ──────────────────────────
        vision_features = []
        for t in range(T):
            feat = self.backbone(x_sequence[:, :, t, :, :])
            vision_features.append(feat)
        x_s = torch.stack(vision_features, dim=2)   # [B, C, T, fH, fW]

        # ── Branch 2: RAFT optical flow (parallel, T-1 pairs) ──────────────
        flows = []
        for t in range(T - 1):
            frame_t   = x_sequence[:, :, t,     :, :]   # [B, 3, H, W]
            frame_t1  = x_sequence[:, :, t + 1, :, :]
            flow = self.raft(frame_t, frame_t1)          # [B, 2, H, W]
            flows.append(flow)

        # ── Warp backbone features using RAFT flows ─────────────────────────
        # Frame 0 has no predecessor — keep it as-is.
        # Frames 1..T-1 are warped using flow from the previous pair.
        x_s_warped_list = [x_s[:, :, 0, :, :]]         # t=0 unchanged
        for t in range(1, T):
            warped = warp_features(x_s[:, :, t, :, :], flows[t - 1])
            x_s_warped_list.append(warped)
        x_s_warped = torch.stack(x_s_warped_list, dim=2)  # [B, C, T, fH, fW]

        # ── MICPL on motion-aligned features ────────────────────────────────
        if training:
            h_s = self.micpl(x_s_warped)
            return x_s + h_s          # residual fusion on original x_s
        else:
            return x_s
