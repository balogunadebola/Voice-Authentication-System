import torch
import torchvision.transforms as transforms
from PIL import Image
import logging
from tqdm import tqdm
from DeepfakeDetection.train import Deep4SNet

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeepfakeDetector:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load model architecture
        self.model = Deep4SNet()
          # Load model weights with weights_only=True for security
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Model loaded from {model_path}")
        logger.info(f"Best validation accuracy: {checkpoint['val_acc']:.2f}%")
        
        # Define transform
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    def predict_single(self, image_path):
        """Predict for a single image"""
        try:
            # Load and transform image
            with Image.open(image_path) as img:
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                image = self.transform(img)
                image = image.unsqueeze(0)  # Add batch dimension
                
            # Predict
            with torch.no_grad():
                image = image.to(self.device)
                outputs = self.model(image)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                pred_prob, predicted = torch.max(probabilities, 1)
                
            # Get results
            is_fake = predicted.item() == 1
            confidence = pred_prob.item() * 100
            
            return {
                'prediction': 'FAKE' if is_fake else 'REAL',
                'confidence': confidence,
                'probabilities': {
                    'real': probabilities[0][0].item() * 100,
                    'fake': probabilities[0][1].item() * 100
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
            return None
        

def main():
    # Initialize detector
    detector = DeepfakeDetector(r"C:/Users/Akorede Balogun/Voice Auth System/DeepfakeDetection/models/best_model.pth")
    
    # Example: Single image prediction
    image_path = r"C:/Users/Akorede Balogun/Voice Auth System/DeepfakeDetection/outputs/hist_user_recording.png"
    result = detector.predict_single(image_path)
    
    if result:
        print(f"\nSingle Image Results for {image_path}:")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2f}%")
        print(f"Probability of Real: {result['probabilities']['real']:.2f}%")
        print(f"Probability of Fake: {result['probabilities']['fake']:.2f}%")


if __name__ == "__main__":
    main()