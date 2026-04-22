import os
import torch
import numpy as np
import cv2

from app.models.model_def import (
    DLA34FeatureExtractor,
    SmallObjectDetector,
    CenterNetHead
)

MODEL_PATH = "app/models/best_micpl_model.pth.zip"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = None
head = None

# Match training config
CHANNELS = 64
SEQ_LEN = 5
NUM_CLASSES = 1


# -------------------------
# Load Model (MATCHES TEST CODE)
# -------------------------
def load_model():
    global model, head

    if not os.path.exists(MODEL_PATH):
        return False

    if model is None:
        checkpoint = torch.load(MODEL_PATH, map_location=device)

        backbone = DLA34FeatureExtractor(out_channels=CHANNELS, pretrained=False)

        model = SmallObjectDetector(
            backbone,
            channels=CHANNELS,
            T=SEQ_LEN
        ).to(device)

        head = CenterNetHead(
            in_channels=CHANNELS,
            num_classes=NUM_CLASSES
        ).to(device)

        model.load_state_dict(checkpoint['model_state_dict'])
        head.load_state_dict(checkpoint['head_state_dict'])

        model.eval()
        head.eval()

    return True


# -------------------------
# Preprocess frame
# -------------------------
def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #  resize to safe dimensions (multiple of 32)
    h, w = frame.shape[:2]

    new_h = (h // 32) * 32
    new_w = (w // 32) * 32

    frame = cv2.resize(frame, (new_w, new_h))

    frame = frame.astype(np.float32) / 255.0

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
    returns: tensor (1, 3, T, H, W)
    """

    processed = []

    for frame in frames:
        frame = preprocess_frame(frame)  # (C, H, W)
        processed.append(frame)

    seq = np.stack(processed, axis=0)   # (T, C, H, W)
    seq = np.transpose(seq, (1, 0, 2, 3))  # (C, T, H, W)
    seq = np.expand_dims(seq, axis=0)   # (1, C, T, H, W)

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
# MAIN INFERENCE
# -------------------------
def run_inference(frames):
    """
    frames: list of OpenCV frames
    returns: list of output frames
    """

    if model is None or head is None:
        raise RuntimeError("Model not loaded")

    output_frames = []

    for i in range(len(frames)):

        if i < SEQ_LEN - 1:
            pad_count = SEQ_LEN - (i + 1)
            seq_frames = [frames[0]] * pad_count + frames[:i + 1]
        else:
            seq_frames = frames[i - SEQ_LEN + 1 : i + 1]

        seq_tensor = build_sequence(seq_frames)

        with torch.no_grad():
            features = model(seq_tensor, training=True)
            features = features[:, :, -1, :, :]
            preds = head(features)

        hm = preds['hm'].detach().cpu().numpy()[0]

        overlay = heatmap_overlay(hm, frames[i])
        output_frames.append(overlay)

    return output_frames