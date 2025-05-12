import os
from AudioRecorder import AudioRecorder
from voiceauth.audioUtils import save_audio

def record_audio_for_user(username):
    """Record audio samples for the given user and save them."""
    base_dir = r'Data'
    user_dir = os.path.join(base_dir, username)

    # Create a new directory for the user if it doesn't exist
    if not os.path.exists(user_dir):
        os.makedirs(user_dir)
        print(f"Created directory for user: {user_dir}")
    else:
        print(f"Directory already exists: {user_dir}")

    # Predefined longer statement about technology
    tech_statement = (
        "Technology has transformed the way we communicate, learn, and interact with the world. "
        "From smartphones to artificial intelligence, it shapes our daily lives and influences our decisions."
    )

    print(f"\nFor the sign-up process, you'll need to record this statement {10} times.")
    print(f"Please read the following statement each time:\n\"{tech_statement}\"")

    # Record multiple samples for the user
    num_samples = 10  # Change this number for more recordings
    recorder = AudioRecorder()
    
    for i in range(num_samples):
        print(f"\nRecording sample {i+1} of {num_samples}")
        input("Press Enter to start recording...")
        print("Recording... Press Enter to stop.")
        recorder.record()
        save_audio(os.path.join(user_dir, f'sample_{i+1}.wav'), recorder.audio_data)
        print(f"Sample {i+1} saved successfully!")
        recorder.audio_data.clear()  # Clear data for next sample

    print("\nAll samples recorded successfully!")
    
    # Save the username to a file for later use
    with open(os.path.join(base_dir, 'last_username.txt'), 'w') as f:
        f.write(username)
        
    # Return the path where audio samples are stored
    return user_dir