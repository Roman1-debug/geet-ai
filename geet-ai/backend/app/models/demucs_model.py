import torch
import torchaudio
from typing import Dict, Optional
import numpy as np

def load_demucs_model(device="cpu"):
    model = torch.hub.load('pytorch/audio', 'demucs', pretrained=True)
    model = model.to(device)
    return model

def separate_stems_with_demucs(
    model,
    device,
    audio: torch.Tensor,
    sample_rate: int = 16000
) -> Optional[Dict[str, torch.Tensor]]:
    try:
        if len(audio.shape) == 1:
            audio = audio.unsqueeze(0)
        if sample_rate != model.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, 
                new_freq=model.sample_rate
            )
            audio = resampler(audio)
        audio = audio / audio.abs().max()
        audio = audio.to(device)
        with torch.no_grad():
            sources = model(audio.unsqueeze(0))
            sources = sources.squeeze(0).cpu()
        stems = {
            "vocals": sources[0],
            "drums": sources[1],
            "bass": sources[2],
            "other": sources[3]
        }
        return stems
    except Exception as e:
        print(f"Error separating stems with Demucs: {e}")
        return None

