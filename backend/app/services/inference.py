import os
import torch
import numpy as np
import cv2

MODEL_PATH = "app/models/best_micpl_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = None
head = None

SEQ_LEN = 5   # from notebook


# -------------------------
# Load Model (Notebook Style)
# -------------------------
def load_model():
    global model, head

    if not os.path.exists(MODEL_PATH):
        return False

    if model is None:
        checkpoint = torch.load(MODEL_PATH, map_location=device)

        # ⚠️ You MUST import these from your model file
        from app.models.model_def import (
            DLA34FeatureExtractor,
            RAFTFlowEstimator,
            SmallObjectDetector,
            CenterNetHead
        )

        CHANNELS = 64
        NUM_CLASSES = 1

        backbone = DLA34FeatureExtractor(out_channels=CHANNELS, pretrained=False)
        raft = RAFTFlowEstimator()
        model = SmallObjectDetector(backbone, raft, channels=CHANNELS, T=SEQ_LEN).to(device)

        head = CenterNetHead(in_channels=CHANNELS, num_classes=NUM_CLASSES).to(device)

        model.load_state_dict(checkpoint['model_state_dict'])
        head.load_state_dict(checkpoint['head_state_dict'])

        model.eval()
        head.eval()

    return True


# -------------------------
# Preprocess single frame
# -------------------------
def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.astype(np.float32) / 255.0

    # Normalize (from notebook)
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    frame = (frame - mean) / std

    frame = np.transpose(frame, (2, 0, 1))  # CHW
    return frame


# -------------------------
# Build sequence tensor
# -------------------------
def build_sequence(frames):
    """
    frames: list of length SEQ_LEN
    returns: tensor (1, C, T, H, W)
    """
    processed = [preprocess_frame(f) for f in frames]

    seq = np.stack(processed, axis=1)  # (C, T, H, W)
    seq = np.expand_dims(seq, axis=0)  # (1, C, T, H, W)

    return torch.tensor(seq, dtype=torch.float32).to(device)


# -------------------------
# Heatmap overlay
# -------------------------
def heatmap_overlay(hm, frame):
    hm = hm.squeeze()

    hm = (hm - hm.min()) / (hm.max() + 1e-8)
    hm = (hm * 255).astype(np.uint8)

    hm = cv2.resize(hm, (frame.shape[1], frame.shape[0]))
    hm = cv2.applyColorMap(hm, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(frame, 0.6, hm, 0.4, 0)
    return overlay


# -------------------------
# Main Inference
# -------------------------
def run_inference(frames):
    """
    frames: list of frames (batch from pipeline)
    returns: list of processed frames
    """

    if model is None or head is None:
        raise RuntimeError("Model not loaded")

    output_frames = []

    # sliding window over frames
    for i in range(len(frames)):
        if i < SEQ_LEN - 1:
            # not enough frames yet → just pass original
            output_frames.append(frames[i])
            continue

        seq_frames = frames[i - SEQ_LEN + 1 : i + 1]

        seq_tensor = build_sequence(seq_frames)

        with torch.no_grad():
            fused = model(seq_tensor)
            final_feat = fused[:, :, -1, :, :]
            preds = head(final_feat)

        hm = preds['hm'].detach().cpu().numpy()[0]

        overlay = heatmap_overlay(hm, frames[i])
        output_frames.append(overlay)

    return output_frames