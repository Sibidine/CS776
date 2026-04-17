import torch
import os

model = None
MODEL_PATH = "app/models/model.pt"

def load_model():
    global model

    if not os.path.exists(MODEL_PATH):
        return False  # ❗ model missing

    if model is None:
        model = torch.load(MODEL_PATH)
        model.eval().cuda()

    return True


def run_inference(frames):
    load_model()

    # preprocess frames → tensor
    # shape: (B, C, H, W)

    with torch.no_grad():
        preds = model(...)  # your input

    # convert predictions → heatmaps (numpy frames)

    return heatmaps