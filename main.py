from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import os
import uuid
from typing import Optional
import asyncio
from music_generation import generate_song, extract_stems

app = FastAPI(title="GeetAI API", description="AI Music Generation API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Create static directory if it doesn't exist
os.makedirs("static", exist_ok=True)

@app.post("/generate")
async def generate_music(
    lyrics: str = Form(...),
    language: str = Form(...),
    genre: str = Form(...),
    mood: str = Form(...),
    tempo: str = Form(...),
    voice_style: str = Form(...),
    voice_cloning: Optional[bool] = Form(False),
    get_stems: Optional[bool] = Form(False)
):
    try:
        # Generate unique ID for this request
        request_id = str(uuid.uuid4())
        
        # Create directory for this request
        output_dir = f"static/{request_id}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate the song asynchronously
        output_path = await asyncio.to_thread(
            generate_song,
            lyrics=lyrics,
            language=language,
            genre=genre,
            mood=mood,
            tempo=tempo,
            voice_style=voice_style,
            voice_cloning=voice_cloning,
            output_dir=output_dir
        )
        
        # Extract stems if requested
        stems_path = None
        if get_stems:
            stems_path = await asyncio.to_thread(
                extract_stems,
                input_path=output_path,
                output_dir=output_dir
            )
        
        # Return the file paths
        return {
            "success": True,
            "song_url": f"/static/{request_id}/song.mp3",
            "stems_url": f"/static/{request_id}/stems/" if stems_path else None
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating music: {str(e)}")

@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = f"static/{filename}"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, media_type='audio/mpeg', filename=filename)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
