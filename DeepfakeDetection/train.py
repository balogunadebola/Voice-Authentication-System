import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import wandb
from tqdm import tqdm
import logging
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
import gc

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HistogramsDataset(Dataset):
    def __init__(self, root_dir, dataset_type='Training_Set', custom_transform=None):
        self.root_dir = Path(root_dir)
        
        # Use exact directory names matching the structure
        if dataset_type not in ['Training_Set', 'Validation_Set', 'Test_Set']:
            raise ValueError(f"Invalid dataset type: {dataset_type}. Must be one of: Training_Set, Validation_Set, Test_Set")
        
        self.dataset_dir = self.root_dir / dataset_type
        logger.info(f"Loading dataset from: {self.dataset_dir}")
        
        # Define image transformations (default or custom)
        self.transform = custom_transform or transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        # Setup paths
        self.real_dir = self.dataset_dir / 'real'
        self.fake_dir = self.dataset_dir / 'fake'
        
        # Get file lists for both PNG and JPG files
        self.real_files = []
        self.fake_files = []
        
        # Image extensions to look for (case-insensitive)
        extensions = ['png', 'jpg', 'jpeg']
        
        # Gather real files
        for ext in extensions:
            patterns = [f'*.{ext.lower()}', f'*.{ext.upper()}']
            for pattern in patterns:
                found_files = list(self.real_dir.glob(pattern))
                self.real_files.extend(found_files)
        
        # Gather fake files
        for ext in extensions:
            patterns = [f'*.{ext.lower()}', f'*.{ext.upper()}']
            for pattern in patterns:
                found_files = list(self.fake_dir.glob(pattern))
                self.fake_files.extend(found_files)
        
        logger.info(f"Found {len(self.real_files)} real and {len(self.fake_files)} fake spectrograms in {dataset_type}")
        
        # Create file paths and labels lists
        self.file_paths = []
        self.labels = []
        
        # Process real files
        for file in self.real_files:
            if self._is_valid_image(file):
                self.file_paths.append(str(file))
                self.labels.append(0)
        
        # Process fake files
        for file in self.fake_files:
            if self._is_valid_image(file):
                self.file_paths.append(str(file))
                self.labels.append(1)
        
        logger.info(f"Successfully loaded {len(self.file_paths)} total valid images")
        
        if len(self.file_paths) == 0:
            raise ValueError(f"No valid images found in {dataset_type} directory")

    def _is_valid_image(self, file_path):
        # Check if the image file is valid
        try:
            with Image.open(file_path) as img:
                img.verify()
            return True
        except Exception as e:
            logger.error(f"Invalid image file {file_path}: {str(e)}")
            return False

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        label = self.labels[idx]
        
        try:
            with Image.open(img_path) as img:
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                image = self.transform(img)
            return image, torch.tensor(label, dtype=torch.long)
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {str(e)}")
            return torch.zeros(1, 128, 128), torch.tensor(label, dtype=torch.long)

class Deep4SNet(nn.Module):
    def __init__(self, num_classes=2):
        super(Deep4SNet, self).__init__()
        
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            
            # Second block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),
            
            # Third block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.4),
            
            # Fourth block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.5),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def train_model():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)

    # Initialize wandb with environment variable
    try:
        print("Initializing wandb...")
        os.environ['WANDB_START_METHOD'] = 'thread'
        os.environ['WANDB_TIMEOUT'] = '600'  # 10 minutes timeout
        wandb.init(
            project="voice-clone-detection",
            config={
                "learning_rate": 0.0003,
                "epochs": 50,
                "batch_size": 32,
                "model": "Deep4SNet",
                "optimizer": "AdamW",
                "weight_decay": 0.01,
                "scheduler": "CosineAnnealingWarmRestarts"
            }
        )
        print("wandb initialized successfully!")
    except Exception as e:
        print(f"Warning: Could not initialize wandb: {str(e)}")
        print("Continuing without wandb logging...")
        wandb.init = lambda *args, **kwargs: None
        wandb.log = lambda *args, **kwargs: None
        wandb.finish = lambda *args, **kwargs: None
    
    # Data augmentation transforms
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Set data directory
    data_dir = "DeepfakeDetection/Data/H-Voice_SiF-Filtered"
    
    # Create datasets
    train_dataset = HistogramsDataset(data_dir, 'Training_Set', train_transform)
    val_dataset = HistogramsDataset(data_dir, 'Validation_Set', val_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    # Initialize model and move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Deep4SNet().to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # Training loop
    best_val_acc = 0.0
    num_epochs = 50
    
    # Ensure model directory exists
    os.makedirs('DeepfakeDetection/models', exist_ok=True)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc='Validation'):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        
        # Update learning rate
        scheduler.step()
        
        # Log metrics
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss / len(train_loader),
            'train_acc': train_acc,
            'val_loss': val_loss / len(val_loader),
            'val_acc': val_acc
        })
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, 'DeepfakeDetection/models/best_model.pth')
            
            print(f'New best model saved! Validation Accuracy: {val_acc:.2f}%')
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%')
    
    wandb.finish()
    print("Training completed!")

if __name__ == "__main__":
    train_model()