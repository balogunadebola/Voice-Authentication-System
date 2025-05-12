# Voice Authentication System with Deepfake Detection

A robust voice authentication system that combines state-of-the-art speaker verification using SpeechBrain with deepfake detection capabilities.

## Features

- **Speaker Verification**: Uses SpeechBrain's ECAPA-TDNN model trained on VoxCeleb for accurate speaker recognition
- **Deepfake Detection**: Implements Deep4SNet model to detect synthetic/cloned voices
- **Voice Activity Detection (VAD)**: Energy-based voice activity detection for better signal processing
- **Two-Factor Verification**: Combines speaker verification with deepfake detection for enhanced security
- **Adaptive Thresholds**: Per-user verification thresholds based on enrollment samples

## System Requirements

- Python 3.11 or higher
- CUDA-capable GPU (optional, for faster processing)
- Audio input device for recording

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd voice-auth-system
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
├── main.py                  # Main application entry point
├── register_user.py         # User registration handling
├── AudioRecorder.py         # Audio recording utilities
├── voiceauth/              # Core voice authentication modules
│   ├── speaker_model.py    # SpeechBrain integration
│   ├── voice_verifier.py   # Main verification logic
│   ├── vad.py             # Voice Activity Detection
│   └── spectral_features.py# Additional voice feature extraction
├── DeepfakeDetection/      # Deepfake detection components
│   ├── DataProcessing.py   # Audio processing for deepfake detection
│   ├── run_record.py       # Deepfake detection inference
│   └── train.py           # Deep4SNet model definition
└── Data/                   # User voice samples storage
```

## Usage

### 1. User Enrollment
```bash
python main.py
# Choose option 1 (Sign Up)
# Follow the prompts to record voice samples
```

### 2. Authentication
```bash
python main.py
# Choose option 2 (Login)
# Read the provided text for verification
```

### 3. Model Retraining
```bash
python main.py
# Choose option 3 (Retrain Model)
# Uses existing voice samples to update the model
```

## Technical Details

### Speaker Verification
- Uses SpeechBrain's ECAPA-TDNN model
- Extracts speaker embeddings for comparison
- Implements cosine similarity for verification
- Adaptive thresholds based on enrollment samples

### Deepfake Detection
- Deep4SNet architecture for synthetic voice detection
- Spectral analysis and histogram-based features
- Confidence threshold of 70% for fake detection

### Voice Processing
- 16kHz sampling rate
- Energy-based VAD for speech detection
- Multi-channel audio support with automatic conversion

## Security Features

1. **Two-Stage Verification**
   - Speaker verification using voice embeddings
   - Deepfake detection to prevent voice cloning attacks

2. **Adaptive Security**
   - Per-user verification thresholds
   - Statistical analysis of enrollment samples
   - Strict confidence requirements for both stages

3. **Signal Processing**
   - Voice activity detection to focus on speech segments
   - Robust feature extraction and normalization
   - Multi-channel audio handling

## Error Handling

The system includes comprehensive error handling and logging:
- Audio recording validation
- Model loading verification
- Feature extraction monitoring
- Verification process logging

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting pull requests.

## License

[Specify your license here]

## Acknowledgments

- SpeechBrain for the speaker recognition model
- PyTorch team for the deep learning framework
- VoxCeleb dataset for model training

## Contact

[Your contact information]
