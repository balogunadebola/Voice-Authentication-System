# SpeechAuth

SpeechAuth is a voice-based authentication system that uses speech recognition and voice embeddings to provide secure access to sensitive operations. It allows users to enroll their voice and authenticate themselves using their unique voiceprint.

## Features

- **Voice Enrollment**: Users can enroll their voice by speaking a passphrase, which is stored as a voiceprint.
- **Voice Authentication**: Authenticate users by comparing their voice against the enrolled voiceprint.
- **Speech-to-Text Commands**: Recognizes spoken commands and processes them.
- **Text-to-Speech Responses**: Provides audio feedback using a text-to-speech engine.
- **Protected Commands**: Ensures sensitive operations like "transfer" or "transaction" are voice-authenticated.

## Requirements

The project requires the following Python libraries:

- `SpeechRecognition==3.10.0`: Main speech-to-text library
- `PyAudio==0.2.13`: Audio input (may need manual installation on some platforms)
- `soundfile==0.12.1`: For reading/writing audio files
- `numpy==1.25.0`: Required for audio processing
- `resemblyzer==0.1.4`: Voice embeddings for speaker recognition
- `openai==0.27.8`: (Optional) For using OpenAI Whisper/API
- `python-dotenv==1.0.0`: (Optional) For managing API keys
- `pyttsx3==2.90`: Text-to-speech conversion
- `pywhatkit==5.4.2`: For sending WhatsApp messages
- `sounddevice==0.4.6`: For real-time audio playback

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd SpeechAuth
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Ensure you have the necessary audio drivers and libraries installed (e.g., eSpeak for text-to-speech on Linux).

## Usage

1. Run the main script:
   ```bash
   python main.py
   ```

2. Choose an option:
   - **Sign Up**: Enroll your voice by speaking a passphrase.
   - **Login**: Authenticate using your voice and execute protected commands.

3. Follow the prompts to complete the desired operation.

## Project Structure

```
SpeechAuth/
├── main.py                # Main entry point of the application
├── requirements.txt       # List of dependencies
├── voice_auth.py          # Voice authentication logic
├── voiceprints/           # Directory for storing voiceprints and audio files
└── README.md              # Project documentation
```

## Notes

- Ensure your microphone is properly configured and accessible.
- For sensitive operations, the system requires voice authentication.
- The project uses `pyttsx3` for text-to-speech, which defaults to the Windows SAPI5 engine.

## Troubleshooting

- **PyAudio Installation Issues**: On some platforms, you may need to install PyAudio manually. For Windows, download the appropriate wheel file from [Unofficial Python Wheels](https://www.lfd.uci.edu/~gohlke/pythonlibs/) and install it using:
  ```bash
  pip install <path-to-wheel-file>
  ```

## License

This project is licensed under the MIT License. See the LICENSE file for details.
