import numpy as np
import wave
from scipy.io import wavfile

def save_audio(filename, audio_data):
    """Saves recorded audio data as a WAV file."""
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(16000)
        wf.writeframes(b''.join(audio_data))

def determine_threshold(log_likelihoods, margin=10.0):
    """Determine an appropriate threshold based on log-likelihoods."""
    # Using the 90th percentile as the baseline threshold
    threshold = np.percentile(log_likelihoods, 90)
    
    # Add a margin to the threshold to ensure it's not too low
    threshold += margin
    
    return threshold