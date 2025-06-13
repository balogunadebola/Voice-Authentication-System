# voice_auth.py
import soundfile as sf
import sounddevice as sd
import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav
from utils import talk

encoder = VoiceEncoder()

def record_phrase(filename, duration=5):
    #print("Please speak your passphrase for enrollment…")
    
    rec = sd.rec(int(16000 * duration), samplerate=16000, channels=1)
    sd.wait()
    sf.write(filename, rec, 16000)
    print(f"Enrollment saved to {filename}")

def enroll_user(phrase_file, embed_file):
    wav = preprocess_wav(phrase_file)
    emb = encoder.embed_utterance(wav)
    np.save(embed_file, emb)
    print("Voiceprint saved to {embed_file}")

def authenticate_user(test_file, enroll_embed,
                      threshold=0.82, duration=5):
    print("Please speak your passphrase to authenticate…")
    talk("Please speak your passphrase to authenticate…")
    rec = sd.rec(int(16000 * duration), samplerate=16000, channels=1)
    sd.wait()
    sf.write(test_file, rec, 16000)

    test_wav = preprocess_wav(test_file)
    test_emb = encoder.embed_utterance(test_wav)
    enrolled_emb = np.load(enroll_embed)

    score = np.dot(enrolled_emb, test_emb) / (
        np.linalg.norm(enrolled_emb) * np.linalg.norm(test_emb)
    )
    print(f"Similarity: {score:.2f}")
    return score >= threshold