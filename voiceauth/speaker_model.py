import torch
import librosa
import numpy as np
import logging
from typing import Dict, Optional, Tuple
import os
from speechbrain.inference.speaker import SpeakerRecognition

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpeakerVerification:
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """Initialize SpeechBrain speaker verification model"""
        self.device = device
        logger.info(f"Initializing SpeechBrain speaker verification model on {device}")
        
        try:
            # Load pretrained SpeechBrain speaker recognition model (ECAPA-TDNN)
            self.model = SpeakerRecognition.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/spkrec-ecapa-voxceleb"
            )
            self.model.to(self.device)
            logger.info("Successfully loaded SpeechBrain model")
        except Exception as e:
            logger.error(f"Error loading SpeechBrain model: {e}")
            raise
        
    def preprocess_audio(self, waveform: np.ndarray) -> torch.Tensor:
        """
        Prepare audio for model inference.
        Handles single/multi-channel audio and ensures correct shape.
        """
        try:
            with torch.no_grad():
                # Convert numpy array to torch tensor
                waveform = torch.from_numpy(waveform).float()
                
                # SpeechBrain expects shape [batch, channel, time]
                if waveform.ndim == 1:
                    waveform = waveform.unsqueeze(0).unsqueeze(0)
                elif waveform.ndim == 2 and waveform.shape[0] > 1:  # Multiple channels
                    waveform = waveform.mean(dim=0, keepdim=True).unsqueeze(0)
                elif waveform.ndim == 2:
                    waveform = waveform.unsqueeze(0)
                return waveform.to(self.device)
        except Exception as e:
            logger.error(f"Error preprocessing audio: {e}")
            raise
            
    def get_embedding(self, waveform: np.ndarray) -> torch.Tensor:
        """Extract speaker embedding using SpeechBrain model"""
        try:
            with torch.no_grad():
                waveform = self.preprocess_audio(waveform)
                embeddings = self.model.encode_batch(waveform)
                return embeddings.squeeze(0).cpu()
        except Exception as e:
            logger.error(f"Error extracting embedding: {e}")
            raise
            
    def verify_speaker(self, enrolled_embedding: torch.Tensor, test_embedding: torch.Tensor, 
                      threshold: float = 0.7) -> Tuple[bool, float]:
        """
        Verify if two embeddings belong to the same speaker using cosine similarity
        """
        try:
            with torch.no_grad():
                # Use SpeechBrain's built-in speaker comparison
                similarity = self.model.similarity(enrolled_embedding.unsqueeze(0), 
                                                test_embedding.unsqueeze(0))
                is_same_speaker = similarity > threshold
                return bool(is_same_speaker), float(similarity)
        except Exception as e:
            logger.error(f"Error during speaker verification: {e}")
            raise
            
    def enroll_speaker(self, waveforms: list) -> torch.Tensor:
        """
        Enroll a speaker using multiple audio samples.
        Returns a mean embedding vector for more robust verification.
        """
        try:
            embeddings = []
            for waveform in waveforms:
                embedding = self.get_embedding(waveform)
                embeddings.append(embedding)
            return torch.mean(torch.stack(embeddings), dim=0)
        except Exception as e:
            logger.error(f"Error during speaker enrollment: {e}")
            raise
        
    @staticmethod
    def load_audio(file_path: str) -> np.ndarray:
        """Load and resample audio file to 16kHz using librosa"""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Audio file not found: {file_path}")
                
            # Load audio file with librosa (automatically handles resampling)
            waveform, _ = librosa.load(file_path, sr=16000, mono=True)
            return waveform
            
        except Exception as e:
            logger.error(f"Error loading audio file {file_path}: {e}")
            raise