from fastapi import APIRouter, UploadFile, File, BackgroundTasks
import uuid
import os

from app.services.pipeline import process_video

router = APIRouter()

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"

task_status = {}

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


@router.post("/upload")
async def upload_video(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    task_id = str(uuid.uuid4())

    input_path = f"{UPLOAD_DIR}/{task_id}.mp4"
    output_path = f"{OUTPUT_DIR}/{task_id}.mp4"

    with open(input_path, "wb") as f:
        f.write(await file.read())

    task_status[task_id] = {"status": "processing"}

    background_tasks.add_task(
        process_video,
        input_path,
        output_path,
        task_status,
        task_id
    )

    return {"task_id": task_id}


@router.get("/result/{task_id}")
def get_result(task_id: str):
    if task_id not in task_status:
        return {"status": "unknown"}

    data = task_status[task_id]

    if data["status"] != "done":
        return {"status": "processing"}

    return {
        "status": "done",
        "video_url": f"/video/{task_id}",
        "error": data.get("error")  # may be None
    }

@router.get("/video/{task_id}")
def serve_video(task_id: str):
    from fastapi.responses import FileResponse
    return FileResponse(f"{OUTPUT_DIR}/{task_id}.mp4", media_type="video/mp4")