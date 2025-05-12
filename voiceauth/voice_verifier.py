import os
import torch
import json
from .speaker_model import SpeakerVerification
from .vad import detect_speech
import numpy as np
import logging
from typing import Dict, List, Optional
import torchaudio

class VoiceVerifier:
    def __init__(self):
        """Initialize the voice verifier with SpeechBrain model"""
        self.speaker_verifier = SpeakerVerification()
        self.enrolled_speakers = {}
        self.thresholds = {}
        
        # Create model directory if it doesn't exist
        os.makedirs(os.path.join('voiceauth', 'model'), exist_ok=True)
        
    def _get_threshold_path(self, username: str) -> str:
        """Get the path for storing speaker threshold"""
        return os.path.join('voiceauth', 'model', f'{username}_threshold.json')
        
    def _get_embedding_path(self, username: str) -> str:
        """Get the path for storing speaker embedding"""
        return os.path.join('voiceauth', 'model', f'{username}_embedding.pt')
        
    def enroll_speaker(self, username: str, audio_directory: str) -> bool:
        """
        Enroll a new speaker using their audio samples
        """
        try:
            # Load all audio samples for the speaker
            waveforms = []
            for file_name in os.listdir(audio_directory):
                if file_name.endswith('.wav'):
                    file_path = os.path.join(audio_directory, file_name)
                    waveform = self.speaker_verifier.load_audio(file_path)
                    
                    # Apply VAD to get speech segments
                    speech_signal = torch.from_numpy(detect_speech(waveform.numpy()[0], 16000))
                    if len(speech_signal) > 0:
                        waveforms.append(speech_signal.unsqueeze(0))
            
            if not waveforms:
                logging.error(f"No valid audio samples found for {username}")
                return False
                
            # Get speaker embedding using SpeechBrain model
            embedding = self.speaker_verifier.enroll_speaker(waveforms)
            
            # Save speaker embedding
            torch.save(embedding, self._get_embedding_path(username))
            self.enrolled_speakers[username] = embedding
            
            # Calculate adaptive threshold using cross-validation
            similarities = []
            for i, wav1 in enumerate(waveforms):
                emb1 = self.speaker_verifier.get_embedding(wav1)
                for j, wav2 in enumerate(waveforms):
                    if i != j:
                        emb2 = self.speaker_verifier.get_embedding(wav2)
                        _, sim = self.speaker_verifier.verify_speaker(emb1, emb2)
                        similarities.append(sim)
            
            # Set threshold as mean - std for security (adjust this based on your security needs)
            threshold = float(np.mean(similarities) - np.std(similarities))
            self.thresholds[username] = threshold
            
            # Save threshold
            with open(self._get_threshold_path(username), 'w') as f:
                json.dump({'threshold': threshold}, f)
                
            logging.info(f"Successfully enrolled speaker {username} with threshold {threshold}")
            return True
            
        except Exception as e:
            logging.error(f"Error enrolling speaker {username}: {e}")
            return False
            
    def verify_speaker(self, username: str, audio_path: str) -> Dict:
        """
        Verify a speaker's identity using a test audio sample
        """
        try:
            if username not in self.enrolled_speakers:
                # Try to load from disk
                embedding_path = self._get_embedding_path(username)
                threshold_path = self._get_threshold_path(username)
                
                if not (os.path.exists(embedding_path) and os.path.exists(threshold_path)):
                    return {'verified': False, 'error': 'Speaker not enrolled'}
                    
                self.enrolled_speakers[username] = torch.load(embedding_path)
                with open(threshold_path, 'r') as f:
                    self.thresholds[username] = json.load(f)['threshold']
            
            # Load and preprocess test audio
            waveform = self.speaker_verifier.load_audio(audio_path)
            speech_signal = torch.from_numpy(detect_speech(waveform.numpy()[0], 16000))
            
            if len(speech_signal) == 0:
                return {'verified': False, 'error': 'No speech detected'}
                
            # Get test embedding
            test_embedding = self.speaker_verifier.get_embedding(speech_signal.unsqueeze(0))
            
            # Get enrolled embedding and threshold
            enrolled_embedding = self.enrolled_speakers[username]
            threshold = self.thresholds.get(username, 0.7)  # Default threshold if not found
            
            # Verify speaker
            is_same_speaker, similarity = self.speaker_verifier.verify_speaker(
                enrolled_embedding, test_embedding, threshold
            )
            
            return {
                'verified': bool(is_same_speaker),
                'confidence': float(similarity),
                'threshold': float(threshold)
            }
            
        except Exception as e:
            logging.error(f"Error verifying speaker {username}: {e}")
            return {'verified': False, 'error': str(e)}
