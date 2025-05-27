import os
import subprocess
from celery import Celery

# Broker & backend on localhost Redis
celery = Celery(
    "worker",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0"
)

VIDEO_DIR = "./storage/videos"
FRAME_DIR = "./storage/frames"

@celery.task
def extract_frames(video_filename: str, fps: int = 10):
    """
    Extract `fps` frames per second from a video.
    """
    input_path = os.path.join(VIDEO_DIR, video_filename)
    name, _ = os.path.splitext(video_filename)
    out_dir = os.path.join(FRAME_DIR, name)
    os.makedirs(out_dir, exist_ok=True)

    # If fps is None or <=0, extract every frame:
    if fps and fps > 0:
        vf_arg = f"fps={fps}"
    else:
        vf_arg = "copy"   # ffmpeg will dump every frame

    cmd = [
        "ffmpeg", "-i", input_path,
        "-vf", vf_arg,
        os.path.join(out_dir, "frame_%06d.jpg")
    ]
    subprocess.run(cmd, check=True)
    return f"Extracted frames to {out_dir}"