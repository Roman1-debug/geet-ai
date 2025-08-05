import os
import torch
import torchaudio
from audiocraft.models import MusicGen
from bark import SAMPLE_RATE, generate_audio, preload_models
from pydub import AudioSegment
import numpy as np
import soundfile as sf
from demucs.pretrained import get_model
from demucs.apply import apply_model
import tempfile

# Load models (in a real app, you might want to do this lazily or with caching)
def load_musicgen_model():
    return MusicGen.get_pretrained('medium')

def load_bark_model():
    preload_models()

# Generate music using MusicGen
def generate_music_from_lyrics(lyrics, genre, mood, tempo, output_path):
    model = load_musicgen_model()
    
    # Set generation parameters based on user input
    model.set_generation_params(
        use_sampling=True,
        top_k=250,
        top_p=0.0,
        temperature=1.0,
        duration=30,  # 30 seconds, can be adjusted
        cfg_coef=3.0,
        # Adjust these based on genre, mood, tempo
    )
    
    # Generate music
    wav = model.generate([lyrics])
    
    # Save the generated music
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torchaudio.save(output_path, wav[0].cpu(), model.sample_rate)
    return output_path

# Generate vocals using Bark
def generate_vocals_from_lyrics(lyrics, language, voice_style, output_path):
    load_bark_model()
    
    # Generate audio from text
    audio_array = generate_audio(lyrics, history_prompt=voice_style)
    
    # Save the generated vocals
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, audio_array, SAMPLE_RATE)
    return output_path

# Combine music and vocals
def combine_music_and_vocals(music_path, vocals_path, output_path):
    # Load audio files
    music = AudioSegment.from_file(music_path)
    vocals = AudioSegment.from_file(vocals_path)
    
    # Adjust volumes as needed
    music = music - 5  # Reduce music volume a bit
    vocals = vocals + 2  # Increase vocals volume a bit
    
    # Combine the tracks
    combined = music.overlay(vocals)
    
    # Export the combined audio
    combined.export(output_path, format="mp3")
    return output_path

# Extract stems using Demucs
def extract_stems(input_path, output_dir):
    # Load the model
    model = get_model('htdemucs')
    
    # Create a temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Apply the model
        sources = apply_model(model, [input_path], device='cpu', split=True, overlap=0.25)
        
        # Save the stems
        stems_dir = os.path.join(output_dir, "stems")
        os.makedirs(stems_dir, exist_ok=True)
        
        stems = ['vocals', 'drums', 'bass', 'other']
        for source, stem in zip(sources[0], stems):
            stem_path = os.path.join(stems_dir, f"{stem}.wav")
            torchaudio.save(stem_path, source.cpu(), model.samplerate)
    
    return stems_dir

# Main function to generate a complete song
def generate_song(lyrics, language, genre, mood, tempo, voice_style, voice_cloning, output_dir):
    # Generate paths
    music_path = os.path.join(output_dir, "music.wav")
    vocals_path = os.path.join(output_dir, "vocals.wav")
    output_path = os.path.join(output_dir, "song.mp3")
    
    # Generate music
    generate_music_from_lyrics(lyrics, genre, mood, tempo, music_path)
    
    # Generate vocals
    generate_vocals_from_lyrics(lyrics, language, voice_style, vocals_path)
    
    # Combine music and vocals
    combine_music_and_vocals(music_path, vocals_path, output_path)
    
    return output_path
