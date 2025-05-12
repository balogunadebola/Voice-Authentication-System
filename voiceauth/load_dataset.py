from datasets import load_dataset
import os
import pickle

# Create directory if it does not exist
os.makedirs("Data", exist_ok=True)

# Step 1: Load the Common Voice dataset with authentication and trust_remote_code
try:
    dataset = load_dataset("mozilla-foundation/common_voice_17_0", "en", split="train", use_auth_token="your_hf_token", trust_remote_code=True)  # Replace with your actual token
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Step 2: Print the first few entries in the dataset
for i in range(5):
    print(dataset[i])

# Step 3: Save the dataset to a local file using pickle
with open("Data/UBM-Dataset.pkl", "wb") as f:
    pickle.dump(dataset, f)

print("Dataset saved successfully as 'UBM-Dataset.pkl'.")