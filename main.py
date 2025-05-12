import os
import logging
from register_user import record_audio_for_user
from voiceauth.voice_verifier import VoiceVerifier
import sounddevice as sd
import wavio
from DeepfakeDetection.DataProcessing import process_audio
from DeepfakeDetection.run_record import DeepfakeDetector

# Configure logging
logging.basicConfig(filename='process.log', level=logging.INFO)

def sign_up(username):
    """
    Sign up a new user by recording their audio and enrolling them in the speaker verification system.
    """
    print(f"Starting sign up for {username}...")

    try:
        # Record and save audio samples in a specified directory for the user
        print("Recording audio for user...")
        audio_directory = record_audio_for_user(username)
        print(f"Audio recorded and saved in directory: {audio_directory}")
        
        # Initialize voice verifier
        voice_verifier = VoiceVerifier()
        
        # Enroll the speaker
        if voice_verifier.enroll_speaker(username, audio_directory):
            print("Speaker enrolled successfully!")
            return True
        else:
            print("Failed to enroll speaker.")
            return False
            
    except Exception as e:
        print(f"An error occurred during the sign-up process: {e}")
        logging.error(f"Error during sign up for {username}: {e}")
        return False

def login(username):
    """
    Login an existing user by recording their audio and verifying their identity.
    """
    try:
        # Create a directory for the login recording
        user_dir = os.path.join("recordings", username)
        os.makedirs(user_dir, exist_ok=True)

        # File path for the audio recording
        output_file = os.path.join(user_dir, f"{username}_recording.wav")

        # Sentence for the user to read
        print("Please read the following sentence clearly: ")
        print("\n\"Technology has transformed the way we communicate, learn, and interact with the world. "
              "From smartphones to artificial intelligence, it shapes our daily lives and influences our decisions.\"\n")

        # Prompt user to start recording
        input("Press Enter to start recording...")

        # Recording settings
        sample_rate = 16000

        # Start recording
        print("Recording... Press Enter again to stop.")
        recording = sd.rec(int(10 * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
        sd.wait()

        # Allow user to stop recording
        input("Recording stopped. Press Enter to save the recording...")

        # Save the recording
        wavio.write(output_file, recording, sample_rate, sampwidth=4)
        print("Recording saved.")

        # Process audio for deepfake detection
        print("Processing the recorded audio for deepfake detection...")
        histogram_path = process_audio(output_file, cutoff_frequency=4000, output_dir=user_dir)
        
        # Deepfake detection
        detector = DeepfakeDetector(r"C:/Users/Akorede Balogun/Voice Auth System/DeepfakeDetection/models/best_model.pth")
        result = detector.predict_single(histogram_path)
        
        if result:
            print(f"\nDeepfake Detection Results:")
            print(f"Prediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']:.2f}%")
            print(f"Probability of Real: {result['probabilities']['real']:.2f}%")
            print(f"Probability of Fake: {result['probabilities']['fake']:.2f}%")
            
            # Only proceed with verification if we're not confident it's fake
            if result['probabilities']['fake'] < 70.0:
                # Initialize voice verifier
                voice_verifier = VoiceVerifier()
                
                # Verify speaker
                verification_result = voice_verifier.verify_speaker(username, output_file)
                
                if verification_result.get('verified', False):
                    print("Authentication Successful")
                    print(f"Confidence: {verification_result.get('confidence', 0):.2f}")
                    return True
                else:
                    print("Authentication Failed")
                    if 'error' in verification_result:
                        print(f"Error: {verification_result['error']}")
                    return False
            else:
                print("Cloned voice detected, authentication failed.")
                return False
                
        return False
        
    except Exception as e:
        print(f"An error occurred during login: {e}")
        logging.error(f"Error during login for {username}: {e}")
        return False

def retrain_user_model(username):
    """
    Retrain a user's speaker verification model using their existing audio samples.
    """
    print(f"Retraining model for user: {username}...")

    try:
        # Use existing audio directory
        audio_directory = os.path.join("Data", username)
        if not os.path.exists(audio_directory):
            print(f"No training data found for user {username}")
            return False
            
        # Initialize voice verifier
        voice_verifier = VoiceVerifier()
        
        # Re-enroll the speaker with existing samples
        if voice_verifier.enroll_speaker(username, audio_directory):
            print("Speaker model retrained successfully!")
            return True
        else:
            print("Failed to retrain speaker model.")
            return False
            
    except Exception as e:
        print(f"An error occurred during model retraining: {e}")
        logging.error(f"Error during retraining for {username}: {e}")
        return False

def main():
    """Prompt user for signup, login, or retrain model."""
    action = input("Choose an option:\n1. Sign Up\n2. Login\n3. Retrain Model\nEnter 1, 2, or 3: ").strip()

    if action == "1":
        username = input("Enter your username to sign up: ").strip()
        print(f"Attempting to sign up with username: {username}")
        sign_up(username)
    elif action == "2":
        username = input("Enter your username to login: ").strip()
        # Check if user exists
        model_dir = os.path.join('voiceauth', 'model')
        if os.path.exists(os.path.join(model_dir, f'{username}_threshold.json')):
            login(username)
        else:
            print("No user found. Please sign up first.")
    elif action == "3":
        username = input("Enter your username to retrain model: ").strip()
        retrain_user_model(username)
    else:
        print("Invalid option. Exiting...")

if __name__ == "__main__":
    print("Starting main execution...")
    main()