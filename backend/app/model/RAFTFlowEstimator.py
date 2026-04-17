# ------------------------------------------------------------------
# RAFTFlowEstimator
# Wraps torchvision's pretrained raft_small as a frozen optical-flow
# branch that runs IN PARALLEL with the DLA-34 backbone.
#
# Input : two consecutive raw RGB frames  [B, 3, H, W]  (values 0-1)
# Output: flow field                      [B, 2, H, W]  (pixels/frame)
# ------------------------------------------------------------------
class RAFTFlowEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
        self.raft = raft_small(weights=Raft_Small_Weights.DEFAULT)
        # Freeze all RAFT weights — used as a pretrained feature extractor only
        self.raft.requires_grad_(False)
        print("RAFT-small loaded (pretrained, frozen).")

    @torch.no_grad()
    def forward(self, frame1, frame2):
        """
        Args:
            frame1, frame2: [B, 3, H, W]  float tensors in [0, 1]
        Returns:
            flow: [B, 2, H, W]  — final flow prediction (last RAFT iteration)
        """
        # torchvision RAFT expects pixel values in [0, 255]
        f1 = (frame1 * 255.0).clamp(0, 255)
        f2 = (frame2 * 255.0).clamp(0, 255)
        # raft returns a list of flow predictions (one per iteration)
        flow_predictions = self.raft(f1, f2)
        return flow_predictions[-1]   # [B, 2, H, W]


def warp_features(x, flow):
    """
    Warp feature map `x` using optical flow `flow`.

    Args:
        x    : [B, C, H, W]  feature map (at feature resolution)
        flow : [B, 2, fH, fW] flow at FULL image resolution

    Returns:
        warped: [B, C, H, W]  — x warped by the downsampled flow
    """
    B, C, H, W = x.shape

    # Downsample flow to match feature map resolution
    flow_down = F.interpolate(flow, size=(H, W), mode='bilinear', align_corners=False)
    # Scale flow values proportionally to the downsampling factor
    scale_h = H / flow.shape[2]
    scale_w = W / flow.shape[3]
    flow_down[:, 0] *= scale_w
    flow_down[:, 1] *= scale_h

    # Build sampling grid
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=x.device, dtype=torch.float32),
        torch.arange(W, device=x.device, dtype=torch.float32),
        indexing='ij'
    )
    grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)  # [1, 2, H, W]
    grid = grid.expand(B, -1, -1, -1)

    # Apply flow to grid and normalise to [-1, 1] for grid_sample
    new_grid = grid + flow_down
    new_grid[:, 0] = 2.0 * new_grid[:, 0] / (W - 1) - 1.0
    new_grid[:, 1] = 2.0 * new_grid[:, 1] / (H - 1) - 1.0
    new_grid = new_grid.permute(0, 2, 3, 1)   # [B, H, W, 2]

    return F.grid_sample(x, new_grid, mode='bilinear',
                         padding_mode='border', align_corners=True)