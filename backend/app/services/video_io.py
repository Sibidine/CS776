import ffmpeg
import os
import cv2


def extract_frames(input_path, frames_dir):
    os.makedirs(frames_dir, exist_ok=True)

    (
        ffmpeg
        .input(input_path)
        .output(f"{frames_dir}/frame_%05d.png")
        .run(overwrite_output=True, quiet=True)
    )


def load_frame_batch(frames_dir, start, batch_size):
    frames = []
    for i in range(start, start + batch_size):
        path = f"{frames_dir}/frame_{i:05d}.png"
        if not os.path.exists(path):
            break
        frame = cv2.imread(path)
        frames.append(frame)
    return frames

def save_output_frames(frames, output_dir, start_idx):
    os.makedirs(output_dir, exist_ok=True)

    for i, frame in enumerate(frames):
        cv2.imwrite(f"{output_dir}/frame_{start_idx + i:05d}.png", frame)

def encode_video(frames_dir, output_path, fps=30):
    (
        ffmpeg
        .input(f"{frames_dir}/frame_%05d.png", framerate=fps)
        .output(
            output_path,
            vcodec="libx264",
            pix_fmt="yuv420p"
        )
        .run(overwrite_output=True, quiet=True)
    )