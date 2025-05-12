import os
import shutil
import numpy as np
from tqdm import tqdm
from DataProcessing import load_audio, filter_audio, ensure_output_directory
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy import signal
import random

def create_spectrogram(audio_data, sr, output_path):
    """Generate and save a spectrogram."""
    plt.figure(figsize=(3, 3))
    plt.specgram(audio_data, Fs=sr)
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def create_modified_audio(audio_data, sr):
    """Create a modified version of the audio for fake samples."""
    modifications = [
        # Time stretching (more variations)
        lambda x: librosa.effects.time_stretch(x, rate=random.uniform(0.7, 1.3)),
        # Pitch shifting (more variations)
        lambda x: librosa.effects.pitch_shift(x, sr=sr, n_steps=random.uniform(-6, 6)),
        # Adding noise (different levels)
        lambda x: x + np.random.normal(0, random.uniform(0.01, 0.03), len(x)),
        # Filtering (different frequencies)
        lambda x: filter_audio(x, random.uniform(1000, 8000), sr),
        # Speed variation
        lambda x: librosa.effects.time_stretch(x, rate=random.uniform(0.9, 1.1)),
        # Reverb effect
        lambda x: np.concatenate([x, 0.5 * x[:-2000], 0.25 * x[:-4000]]),
    ]
    
    # Apply 2-4 random modifications
    num_mods = random.randint(2, 4)
    selected_mods = random.sample(modifications, num_mods)
    
    modified = audio_data
    for mod in selected_mods:
        modified = mod(modified)
    return modified

def prepare_dataset(source_dir, output_base_dir, split_ratios=(0.7, 0.15, 0.15)):
    """Prepare the dataset for training."""
    # Create dataset directories
    datasets = ['Training_Set', 'Validation_Set', 'Test_Set']
    for dataset in datasets:
        for label in ['real', 'fake']:
            os.makedirs(os.path.join(output_base_dir, dataset, label), exist_ok=True)
    
    # Get all wav files
    wav_files = []
    for root, _, files in os.walk(source_dir):
        wav_files.extend([os.path.join(root, f) for f in files if f.endswith('.wav')])
    
    # Shuffle files
    random.shuffle(wav_files)
    
    # Calculate splits
    n_files = len(wav_files)
    n_train = int(n_files * split_ratios[0])
    n_val = int(n_files * split_ratios[1])
    
    splits = {
        'Training_Set': wav_files[:n_train],
        'Validation_Set': wav_files[n_train:n_train + n_val],
        'Test_Set': wav_files[n_train + n_val:]
    }
    
    # Process each split
    for split_name, files in splits.items():
        print(f"\nProcessing {split_name}...")
        for file_path in tqdm(files):
            # Load audio
            audio_data, sr = load_audio(file_path)
            
            # Create real sample
            output_path = os.path.join(output_base_dir, split_name, 'real', 
                                     f"{os.path.splitext(os.path.basename(file_path))[0]}_real.png")
            create_spectrogram(audio_data, sr, output_path)
            
            # Create fake samples (4 per real sample instead of 2)
            for i in range(4):
                modified = create_modified_audio(audio_data, sr)
                output_path = os.path.join(output_base_dir, split_name, 'fake',
                                         f"{os.path.splitext(os.path.basename(file_path))[0]}_fake_{i}.png")
                create_spectrogram(modified, sr, output_path)

if __name__ == "__main__":
    # Initialize wandb
    os.environ['WANDB_MODE'] = 'disabled'  # Disable wandb for data preparation
    
    source_dir = "Data/Ade"  # Directory with real voice samples
    output_dir = "DeepfakeDetection/Data/H-Voice_SiF-Filtered"
    prepare_dataset(source_dir, output_dir)
    print("Dataset preparation completed!")
