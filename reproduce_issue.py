
import os
import soundfile as sf
import numpy as np
import traceback
from scipy import signal
import torch

try:
    print("Loading test.wav...")
    audio_path = "test.wav"
    audio_data, sr_native = sf.read(audio_path)
    print(f"Audio loaded. Shape: {audio_data.shape}, Dtype: {audio_data.dtype}, SR: {sr_native}")

    if audio_data.ndim > 1:
        print(f"Audio is multidimensional (Values: {audio_data.shape[1] if len(audio_data.shape) > 1 else '?'})")
    
    # Simulate the backend logic
    target_sr = 16000
    if sr_native != target_sr:
        print(f"Resampling from {sr_native} to {target_sr}...")
        num_samples = int(len(audio_data) * target_sr / sr_native)
        audio_data = signal.resample(audio_data, num_samples)
        sr = target_sr
        print(f"Resampled. New Shape: {audio_data.shape}")
        
    # Check if Whisper would crash
    print("Checking for Whisper compatibility...")
    if audio_data.ndim > 1:
        print("❗ WARNING: Audio is not mono! This matches the hypothesis.")
    
    # Try to load Whisper and run log_mel_spectrogram manually (lighter than full transcribe)
    try:
        import whisper
        from whisper.audio import log_mel_spectrogram, pad_or_trim
        
        # Convert to float32 if not already
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
            
        # If I don't flatten it, does it crash?
        print("Attempting log_mel_spectrogram with current data...")
        # Simulating what happens inside whisper
        # Whisper expects torch tensor
        audio_tensor = torch.from_numpy(audio_data)
        
        # If it's stereo, we need to mix to mono, which the backend MIGHT be missing
        print(f"Tensor shape passed to Whisper: {audio_tensor.shape}")
        
        # This is where it likely crashes if shape is wrong
        mel = log_mel_spectrogram(audio_tensor)
        print("log_mel_spectrogram successful!")
        
    except Exception as e:
        print(f"❌ Crash reproduced within script: {e}")
        traceback.print_exc()

except Exception as e:
    print(f"General error: {e}")
    traceback.print_exc()
