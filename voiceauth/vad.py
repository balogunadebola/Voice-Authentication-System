import numpy as np

def detect_speech(audio, sampling_rate, frame_duration=0.02, min_speech_energy=0.01):
    """
    Perform Voice Activity Detection (VAD) using energy-based method.
    
    Args:
        audio: Input audio signal
        sampling_rate: Sampling rate of the audio
        frame_duration: Duration of each frame in seconds
        min_speech_energy: Minimum energy threshold for speech
    
    Returns:
        speech_frames: Audio signal with only speech frames
    """
    # Calculate frame size
    frame_size = int(frame_duration * sampling_rate)
    
    # Split audio into frames
    n_frames = len(audio) // frame_size
    frames = np.array_split(audio[:n_frames * frame_size], n_frames)
    
    # Calculate energy for each frame
    energies = np.array([np.sum(frame**2) / len(frame) for frame in frames])
    
    # Normalize energies
    energies = energies / np.max(energies)
    
    # Find speech frames
    speech_mask = energies > min_speech_energy
    
    # Concatenate speech frames
    speech_frames = np.concatenate([frames[i] for i in range(len(frames)) if speech_mask[i]])
    
    return speech_frames
