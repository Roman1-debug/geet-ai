from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uuid
import os
import torch

from app.models.bark_model import load_bark_model, generate_vocals_with_bark
from app.models.demucs_model import load_demucs_model, separate_stems_with_demucs
from app.services.storage import save_generated_song, get_song_path, cleanup_old_files

# Initialize FastAPI app
app = FastAPI()

# Load models at startup
device = "cuda" if torch.cuda.is_available() else "cpu"
bark_model, bark_processor = load_bark_model(device=device)
demucs_model = load_demucs_model(device=device)

# Cleanup old files on startup
cleanup_old_files("static/songs")

@app.post("/generate/")
async def generate_song(
    lyrics: str = Form(...),
    genre: str = Form(...),
    mood: str = Form(...),
    tempo: int = Form(...),
    vocal_style: str = Form(...)
):
    song_id = str(uuid.uuid4())
    output_path = get_song_path(song_id, "final.wav")

    vocals = generate_vocals_with_bark(
        model=bark_model,
        processor=bark_processor,
        device=device,
        lyrics=lyrics,
        voice_preset=vocal_style
    )

    if vocals is None:
        return {"error": "Failed to generate vocals"}

    save_generated_song(vocals, output_path)
    return {"song_id": song_id, "url": f"/songs/{song_id}/final.wav"}

@app.get("/songs/{song_id}/{filename}")
def get_song(song_id: str, filename: str):
    path = get_song_path(song_id, filename)
    if not os.path.exists(path):
        return {"error": "File not found"}
    return FileResponse(path, media_type="audio/wav")
 
