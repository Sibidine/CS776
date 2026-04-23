import os
import ffmpeg
import subprocess
import shutil

from app.services.video_io import (
    extract_frames,
    load_frame_batch,
    save_output_frames,
    encode_video
)
from app.services.inference import run_inference,load_model

BATCH_SIZE = 8


def get_fps(input_path):
    probe = ffmpeg.probe(input_path)
    video_stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    return eval(video_stream['r_frame_rate'])  # e.g. "30/1"


def process_video(input_path, output_path, status_dict, task_id):
    model_available = load_model()

    if not model_available:
        # fallback: copy original video
        shutil.copy(input_path, output_path)

        status_dict[task_id] = {
            "status": "done",
            "error": "Model not found. Showing original video."
        }
        return
    base = os.path.splitext(os.path.basename(input_path))[0]

    frames_dir = f"tmp/{base}_frames"
    output_frames_dir = f"tmp/{base}_out"

    # 1. Extract frames
    extract_frames(input_path, frames_dir)

    fps = get_fps(input_path)

    idx = 1
    out_idx = 1

    while True:
        batch = load_frame_batch(frames_dir, idx, BATCH_SIZE)
        if not batch:
            break

        heatmaps = run_inference(batch)

        save_output_frames(heatmaps, output_frames_dir, out_idx)

        idx += BATCH_SIZE
        out_idx += len(heatmaps)

    # 2. Encode final video
    encode_video(output_frames_dir, output_path, fps)
    
    status_dict[task_id] = {
    "status": "done",
    "error": None
}
    shutil.rmtree(frames_dir, ignore_errors=True)
    shutil.rmtree(output_frames_dir, ignore_errors=True)