import numpy as np
import logging
from sklearn.mixture import GaussianMixture
import joblib
from scipy.io import wavfile
from feature_extraction import extract_features  # Importing the feature extraction function
from tqdm import tqdm  # Import tqdm for progress bar
import time
from multiprocessing import Pool
import os

# Configure logging for the main script as well (if not already done)
logging.basicConfig(filename='process.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_file(file_path):
    """Process a single WAV file and extract features."""
    try:
        rate, audio = wavfile.read(file_path)
        logging.info(f"Successfully read file: {os.path.basename(file_path)} with sample rate: {rate}")
        
        features = extract_features(audio, rate)
        
        if features is not None:
            logging.info(f"Features extracted from {os.path.basename(file_path)}: shape {features.shape}")
        
        return features
    
    except Exception as e:
        logging.error(f"Error reading file {file_path}: {e}")
        return None

def load_features_from_directory(directory):
    """Load all extracted features from WAV files in the specified directory."""
    all_features = []
    
    # Get all wav files from all user directories
    wav_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.wav'):
                wav_files.append(os.path.join(root, file))
    
    if not wav_files:
        logging.error(f"No WAV files found in {directory}")
        return np.array([])
    
    logging.info(f"Found {len(wav_files)} WAV files")
    
    # Process files with progress bar
    for file_path in tqdm(wav_files, desc="Processing audio files"):
        features = process_file(file_path)
        if features is not None and features.size > 0:
            all_features.append(features)
    
    if not all_features:
        return np.array([])
    
    return np.vstack(all_features)

def train_ubm(features, n_components=32, max_iter=200, patience=10):
    """Train a Universal Background Model (UBM) using GMM with early stopping."""
    logging.info("Initializing Gaussian Mixture Model for UBM training.")
    
    ubm_model = GaussianMixture(
        n_components=n_components,
        covariance_type='diag',
        max_iter=max_iter,
        n_init=5,  # Try multiple initializations
        reg_covar=1e-6  # Add small regularization
    )
    
    logging.info("Fitting the UBM model to the extracted features...")
    
    start_time = time.time()
    best_log_likelihood = -np.inf
    no_improvement_count = 0
    
    ubm_model.fit(features)
    
    elapsed_time = time.time() - start_time
    logging.info(f"UBM model training completed successfully in {elapsed_time:.2f} seconds.")
    
    return ubm_model

if __name__ == "__main__":
    # Use both the Data directory and recordings directory for UBM training
    data_directories = ["Data", "recordings"]
    all_features = []
    
    for directory in data_directories:
        if os.path.exists(directory):
            features = load_features_from_directory(directory)
            if features.size > 0:
                all_features.append(features)
                logging.info(f"Loaded features from {directory}: {features.shape}")
    
    if not all_features:
        logging.error("No features loaded. Exiting.")
        exit(1)
    
    # Combine all features
    all_features = np.vstack(all_features)
    logging.info(f"Total features loaded: {all_features.shape}")
    
    # Train UBM
    n_components = 32
    logging.info("Starting UBM training...")
    
    ubm_model = train_ubm(all_features, n_components)
    
    # Save the trained UBM model
    model_dir = 'voiceauth/model'
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'ubm_model.pkl')
    
    try:
        joblib.dump(ubm_model, model_path)
        logging.info(f"UBM model trained and saved successfully at {model_path}")
    except Exception as e:
        logging.error(f"Error saving UBM model: {e}")