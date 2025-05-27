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
def extract_frames(video_filename, fps=1):
    """
    Extract frames from a video at `fps` frames per second.
    """
    input_path = os.path.join(VIDEO_DIR, video_filename)
    # make a subfolder for frames of this video
    name, _ = os.path.splitext(video_filename)
    out_dir = os.path.join(FRAME_DIR, name)
    os.makedirs(out_dir, exist_ok=True)

    # ffmpeg command: e.g. 1 fps -> one image per second
    cmd = [
        "ffmpeg", "-i", input_path,
        "-vf", f"fps={fps}",
        os.path.join(out_dir, "frame_%05d.jpg")
    ]
    subprocess.run(cmd, check=True)
    return f"Extracted frames to {out_dir}"
