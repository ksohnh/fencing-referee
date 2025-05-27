from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import shutil, uuid, os, json
from tasks import extract_frames

from models import AnnotationBundle

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

VIDEO_DIR = "./storage/videos"
FRAME_DIR = "./storage/frames"
ANNOT_FILE = "./annotations.json"

os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(FRAME_DIR, exist_ok=True)
if not os.path.exists(ANNOT_FILE):
    with open(ANNOT_FILE, "w") as f:
        json.dump({}, f)

@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    # 1. Generate a unique ID for this video
    vid_id = str(uuid.uuid4())

    # 2. Compute the extension and full filename
    ext = os.path.splitext(file.filename)[1]
    saved_filename = vid_id + ext

    # 3. Save the incoming file to disk
    path = os.path.join(VIDEO_DIR, saved_filename)
    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 4. Enqueue the frame-extraction job for 10 fps by default…
    extract_frames.delay(saved_filename, fps=10)
    # 5. Return the ID & original filename
    return {"video_id": vid_id, "filename": file.filename}

@app.get("/videos")
def list_videos():
    return os.listdir(VIDEO_DIR)

@app.post("/annotations/{video_id}")
def save_annotations(video_id: str, ann: AnnotationBundle):
    with open(ANNOT_FILE, "r+") as f:
        data = json.load(f)
        data[video_id] = ann.dict()
        f.seek(0)
        json.dump(data, f, indent=2)
        f.truncate()
    return {"status": "saved"}

@app.get("/annotations/{video_id}")
def load_annotations(video_id: str):
    with open(ANNOT_FILE) as f:
        data = json.load(f)
    return data.get(video_id, {})

app.mount("/videos", StaticFiles(directory=VIDEO_DIR), name="videos")
