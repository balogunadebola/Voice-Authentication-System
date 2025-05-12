import os
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import scipy.signal

def ensure_output_directory(dir_name):
    """
    Ensure that the specified directory exists.
    
    Args:
        dir_name (str): Directory name to check/create.
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def load_audio(file_path, sr=44100):
    """
    Load an audio file with the specified sample rate.
    
    Args:
        file_path (str): Path to the audio file.
        sr (int): Sample rate.
        
    Returns:
        tuple: Audio time series and sample rate.
    """
    audio, sr = librosa.load(file_path, sr=sr)
    return audio, sr

def filter_audio(audio_data, cutoff_frequency, sr):
    """
    Apply a low-pass filter to the audio data.
    
    Args:
        audio_data (np.ndarray): Audio data.
        cutoff_frequency (float): Cutoff frequency for the low-pass filter.
        sr (int): Sample rate of the audio.
        
    Returns:
        np.ndarray: Filtered audio data.
    """
    cutoff_frequency = float(cutoff_frequency)
    
    nyquist_frequency = sr / 2
    cutoff_normalized = cutoff_frequency / nyquist_frequency
    b, a = scipy.signal.butter(4, cutoff_normalized, btype='low')
    filtered_audio = scipy.signal.filtfilt(b, a, audio_data)
    return filtered_audio

def plot_spectrogram(audio, sr, title, output_dir):
    """
    Plot and save the spectrogram of the audio.
    
    Args:
        audio (np.ndarray): Audio data.
        sr (int): Sample rate of the audio.
        title (str): Title for the spectrogram plot.
        output_dir (str): Directory to save the spectrogram image.
    """
    ensure_output_directory(output_dir)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max), sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{title.replace(' ', '_').lower()}.png")
    plt.savefig(save_path)
    plt.close()

def compute_histogram(filtered_audio, file_name, output_dir):
    """
    Compute and save the histogram of filtered audio data.
    
    Args:
        filtered_audio (np.ndarray): Filtered audio data.
        file_name (str): Name of the audio file.
        output_dir (str): Directory to save the histogram image.
        
    Returns:
        str: Path to the saved histogram image.
    """
    ensure_output_directory(output_dir)
    hist, bins = np.histogram(filtered_audio, bins=256, range=(-1, 1))
    plt.figure()
    plt.bar(bins[:-1], hist, width=(bins[1] - bins[0]), color='black')
    save_path = os.path.join(output_dir, f"hist_{file_name}.png")
    plt.savefig(save_path)
    plt.close()
    return save_path

def process_audio(file_path, cutoff_frequency=4000, output_dir=None):
    """
    Process the audio by filtering, plotting spectrograms, and computing histograms.
    
    Args:
        file_path (str): Path to the audio file.
        cutoff_frequency (float): Cutoff frequency for the low-pass filter.
        output_dir (str, optional): Directory to save the spectrogram and histogram images.
        
    Returns:
        str: Path to the saved histogram image.
    """
    # If no output directory is provided, use the directory of the audio file
    if output_dir is None:
        output_dir = os.path.dirname(file_path)
    
    # Ensure the output directory exists
    ensure_output_directory(output_dir)
    
    # Load audio
    audio, sr = load_audio(file_path, sr=44100)
    
    # Plot and save original spectrogram
    plot_spectrogram(audio, sr, title='Original Audio Spectrogram', output_dir=output_dir)
    
    # Filter audio
    filtered_audio = filter_audio(audio, cutoff_frequency, sr)
    
    # Plot and save filtered spectrogram
    plot_spectrogram(filtered_audio, sr, title='Filtered Audio Spectrogram', output_dir=output_dir)
    
    # Compute and save histogram
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    histogram_path = compute_histogram(filtered_audio, file_name, output_dir=output_dir)
    
    # Return the histogram file path
    return histogram_path

# Main execution
if __name__ == "__main__":
    # Set the cutoff frequency and directory for saving outputs
    cutoff_frequency = 4000  # Example fixed cutoff frequency
    user_dir = "processed_audio"  # Specify the directory where results will be saved
    
    # Replace with the path to your recorded audio file
    output_file = "path_to_your_audio_file.wav"
    
    # Process the audio automatically
    histogram_path = process_audio(output_file, cutoff_frequency=cutoff_frequency, output_dir=user_dir)
    print(f"Histogram saved at: {histogram_path}")

    