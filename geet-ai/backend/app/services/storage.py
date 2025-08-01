import os
import shutil
import wave
from datetime import datetime, timedelta
import numpy as np
import torch
import torchaudio

def save_generated_song(audio_data: torch.Tensor, file_path: str):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if isinstance(audio_data, torch.Tensor):
        audio_data = audio_data.detach().cpu().numpy()
    if len(audio_data.shape) > 2:
        audio_data = audio_data.squeeze()
    if audio_data.dtype != np.int16:
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data))
        audio_data = (audio_data * 32767).astype(np.int16)
    torchaudio.save(
        file_path, 
        torch.from_numpy(audio_data).float(), 
        16000
    )

def get_song_path(song_id: str, filename: str) -> str:
    return os.path.join("static", "songs", song_id, filename)

def cleanup_old_files(directory: str, hours: int = 24):
    now = datetime.now()
    cutoff_time = now - timedelta(hours=hours)
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
            if file_mtime < cutoff_time:
                try:
                    os.remove(file_path)
                    print(f"Removed old file: {file_path}")
                except Exception as e:
                    print(f"Error removing file {file_path}: {e}")
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            try:
                if not os.listdir(dir_path):
                    os.rmdir(dir_path)
                    print(f"Removed empty directory: {dir_path}")
            except Exception as e:
                print(f"Error removing directory {dir_path}: {e}")
 
