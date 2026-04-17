import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# -----------------------------------
# DLA BACKBONE
# -----------------------------------
class DLA34FeatureExtractor(nn.Module):
    def __init__(self, out_channels=64, pretrained=True, freeze_backbone=True):
        super().__init__()

        self.dla = timm.create_model(
            "hf_hub:timm/dla34.in1k",
            pretrained=pretrained,
            features_only=True,
            out_indices=(1, 2, 3)
        )

        if freeze_backbone:
            self.dla.requires_grad_(False)

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

    def forward(self, x):
        with torch.no_grad():
            f1, f2, f3 = self.dla(x)

        target_size = f2.shape[-2:]

        p3 = F.interpolate(self.up_c3(f3), size=target_size, mode='bilinear', align_corners=False)
        p2 = self.up_c2(f2)
        p1 = F.avg_pool2d(self.up_c1(f1), 2)

        return self.fuse(p1 + p2 + p3)


# -----------------------------------
# MICPL MODULE (⚠️ REQUIRED)
# -----------------------------------
import torch.nn.functional as F


# -----------------------------------
# MPM NODE (ConvLSTM-style)
# -----------------------------------
class MPM_Node(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super().__init__()
        self.padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            padding=self.padding
        )

    def forward(self, x, h_prev, c_prev):
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)

        i, f, o, g = torch.split(gates, gates.size(1) // 4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


# -----------------------------------
# MOTION-VISION ADAPTER
# -----------------------------------
class MotionVisionAdapter(nn.Module):
    def __init__(self, channels, T):
        super().__init__()
        self.T = T

        self.relation_conv = nn.Conv2d(2 * channels, channels // T, kernel_size=1)
        self.adapting_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, h_tilde, h_sequence):
        batch, c, h, w = h_tilde.size()

        outputs = []

        for t in range(self.T):
            h_t = h_sequence[:, :, t, :, :]

            combined = torch.cat([h_tilde, h_t], dim=1)

            relation_weight = F.relu(self.relation_conv(combined))

            adapted = self.adapting_conv(
                h_t * relation_weight.mean(dim=1, keepdim=True)
            )

            outputs.append(adapted)

        return torch.stack(outputs, dim=2)


# -----------------------------------
# MICPL MODULE (CORRECT)
# -----------------------------------
class MICPL_Module(nn.Module):
    def __init__(self, channels, T, num_layers=2):
        super().__init__()

        self.T = T
        self.num_layers = num_layers

        self.mpm_layers = nn.ModuleList([
            MPM_Node(channels, channels) for _ in range(num_layers)
        ])

        self.mva = MotionVisionAdapter(channels, T)

    def forward(self, x_s):
        """
        x_s: [B, C, T, H, W]
        """

        batch, channels, T, height, width = x_s.size()

        # initialize hidden states
        h = [
            torch.zeros(batch, channels, height, width, device=x_s.device)
            for _ in range(self.num_layers)
        ]
        c = [
            torch.zeros(batch, channels, height, width, device=x_s.device)
            for _ in range(self.num_layers)
        ]

        layer_outputs = []

        # -------- MPM pass --------
        for t in range(T):
            current_input = x_s[:, :, t, :, :]

            for l in range(self.num_layers):
                h[l], c[l] = self.mpm_layers[l](current_input, h[l], c[l])
                current_input = h[l]

            layer_outputs.append(h[-1])

        # [B, C, T, H, W]
        h_s = torch.stack(layer_outputs, dim=2)

        # -------- MVA refinement --------
        h_tilde = h_s[:, :, -1, :, :]
        motion_patterns = self.mva(h_tilde, h_s)

        return motion_patterns


# -----------------------------------
# MAIN MODEL
# -----------------------------------
class SmallObjectDetector(nn.Module):
    def __init__(self, backbone, channels=64, T=5):
        super().__init__()
        self.backbone = backbone
        self.micpl = MICPL_Module(channels, T)

    def forward(self, x_sequence, training=True):
        """
        x_sequence: [B, 3, T, H, W]
        """

        vision_features = []

        for t in range(x_sequence.size(2)):
            feat = self.backbone(x_sequence[:, :, t, :, :])
            vision_features.append(feat)

        x_s = torch.stack(vision_features, dim=2)  # [B, C, T, H/4, W/4]

        if training:
            h_s = self.micpl(x_s)
            return x_s + h_s
        else:
            return x_s


# -----------------------------------
# CENTERNET HEAD
# -----------------------------------
class CenterNetHead(nn.Module):
    def __init__(self, in_channels, num_classes=1):
        super().__init__()

        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_classes, 1)
        )

        self.wh_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 2, 1)
        )

        self.reg_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 2, 1)
        )

        # Bias init (important!)
        self.cls_head[-1].bias.data.fill_(-2.19)

    def forward(self, x):
        hm = torch.sigmoid(self.cls_head(x))
        wh = self.wh_head(x)
        reg = self.reg_head(x)

        return {'hm': hm, 'wh': wh, 'reg': reg}