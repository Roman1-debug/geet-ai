import torch
from transformers import BarkProcessor, BarkModel
import numpy as np
from typing import Optional

def load_bark_model(device="cpu"):
    processor = BarkProcessor.from_pretrained("suno/bark-small")
    model = BarkModel.from_pretrained("suno/bark-small")
    model = model.to(device)
    return model, processor

def generate_vocals_with_bark(
    model,
    processor,
    device,
    lyrics: str,
    voice_preset: str = "v2/en_speaker_6"
) -> Optional[torch.Tensor]:
    try:
        inputs = processor(lyrics, voice_preset=voice_preset)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            audio_array = model.generate(**inputs)
        audio_tensor = audio_array.cpu()
        if audio_tensor.shape[0] > 1:
            audio_tensor = torch.mean(audio_tensor, dim=0)
        return audio_tensor
    except Exception as e:
        print(f"Error generating vocals with Bark: {e}")
        return None
 
