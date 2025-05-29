# alexa.py
import speech_recognition as sr
import pyttsx3
import pywhatkit
import datetime
import os

# … your other imports …
from voice_auth import authenticate_user, record_phrase, enroll_user

listener = sr.Recognizer()
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)
# …

def talk(text):
    engine.say(text)
    engine.runAndWait()

def take_command():
    try:
        with sr.Microphone() as source:
            print("Listening…")
            voice = listener.listen(source)
            command = listener.recognize_google(voice)
            command = command.lower()
            if 'alat' in command:
                command = command.replace('alat', '')
            return command
    except Exception as e:
        print(f"Error: {e}")
        return ""

def signup():
    username = input("Enter your username for sign up: ").strip()
    os.makedirs("voiceprints", exist_ok=True)

    #Define file names with username prefix
    phrase_file = f"voiceprints/{username}_enroll.wav"
    voiceprint_file = f"voiceprints/{username}_voiceprint.npy"

    print("Please speak your passphrase for enrollment…")
    record_phrase(phrase_file) # save the recording to a specific file

    enroll_user(phrase_file, voiceprint_file)

    print(f"Enrollment saved for {username}.\nSign up complete! Please restart and log in.")
    return

def login(username):
    protected = ["transfer", "send","transaction", "balance", "pay", "open account"]
    while True:
        command = take_command()
        print("Heard:", command)

        if any(p in command for p in protected):

            enrolled_file = f"voiceprints/{username}_voiceprint.npy"
            test_file = f"voiceprints/{username}_test.wav"

            if not authenticate_user(test_file=test_file, enroll_embed=enrolled_file):
                print("Authentication failed. Please try again or use your password.")
                talk("Authentication failed. Please try again or use your password.")
                return False    
            
        if "transfer" in command or "send" in command:
            print("Processing your funds transfer.")
            talk("Processing your funds transfer.")
            # call your banking-API here…
            break

        else:
            print("Sorry, I didn't understand the protected action.")
            talk("Sorry, I didn't understand the protected action.")
            continue



if __name__ == "__main__":
    print("Welcome! Please choose:")
    print("1. Sign Up (New User)")
    print("2. Login (Existing User)")

    choice = input("Enter 1 or 2: ").strip()
    if choice == "1":
        signup()
    elif choice == "2":
        username = input("Enter your username: ").strip()
        enrolled_file = f"voiceprints/{username}_voiceprint.npy"
        if not os.path.exists(enrolled_file):
            print(f"No enrollment found for user {username}. Please sign up first.")
        else:
            login(username)
    else:
        print("Invalid choice. Exiting.")
