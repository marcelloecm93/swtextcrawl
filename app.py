import threading
import uuid
from pathlib import Path

import shutil
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from audio import get_duration
from renderer import render

MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB

UPLOADS = Path("uploads")
OUTPUT = Path("output")
UPLOADS.mkdir(exist_ok=True)
OUTPUT.mkdir(exist_ok=True)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# job_id → {status, output, error}
jobs: dict[str, dict] = {}


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.post("/generate")
async def generate(
    episode_label: str = Form(""),
    episode_title: str = Form(""),
    body_text: str = Form(""),
    audio: UploadFile = ...,
):
    if not body_text.strip():
        raise HTTPException(status_code=400, detail="Crawl body text is required.")

    job_id = uuid.uuid4().hex[:8]
    suffix = Path(audio.filename or "audio.mp3").suffix.lower()
    if suffix not in {".mp3", ".wav", ".ogg", ".m4a"}:
        raise HTTPException(status_code=400, detail="Unsupported audio format.")

    audio_path = UPLOADS / f"{job_id}{suffix}"
    written = 0
    with audio_path.open("wb") as f:
        while chunk := await audio.read(1024 * 256):
            written += len(chunk)
            if written > MAX_UPLOAD_BYTES:
                audio_path.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="Audio file exceeds 50 MB limit.")
            f.write(chunk)

    try:
        duration = get_duration(audio_path)
    except Exception as exc:
        audio_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=f"Cannot read audio duration: {exc}")

    warnings = []
    if duration < 15:
        warnings.append("Audio is very short — crawl may scroll too fast to read.")

    output_path = OUTPUT / f"crawl_{job_id}.mp4"
    jobs[job_id] = {"status": "rendering", "warnings": warnings}

    def _run():
        try:
            render(episode_label, episode_title, body_text, audio_path, output_path, duration)
            jobs[job_id]["status"] = "done"
            jobs[job_id]["output"] = str(output_path)
        except Exception as exc:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"] = str(exc)

    threading.Thread(target=_run, daemon=True).start()
    return {"job_id": job_id, "warnings": warnings}


@app.get("/status/{job_id}")
def status(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job


@app.get("/download/{job_id}")
def download(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job["status"] != "done":
        raise HTTPException(status_code=409, detail="Render not complete.")
    return FileResponse(
        job["output"],
        media_type="video/mp4",
        filename=f"crawl_{job_id}.mp4",
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=3000, reload=False)
