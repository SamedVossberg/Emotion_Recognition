from matplotlib import pyplot as plt, transforms
import torch
import torch.nn as nn
from torchvision.transforms import ToTensor
from efficientnet_pytorch import EfficientNet
from models.affect_model import AffectModel
from PIL import Image

import torch
from torchvision import transforms
from torchvision.datasets import DatasetFolder
from torch.utils.data import DataLoader
import torchvision


model = AffectModel("affectNet_emotion_model_best_v2.pth")

# Load FER dataset and create DataLoader

# Define the transformation
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Path to the FER dataset
# fer_dataset_path = 'dataset/fer/test' 
fer_dataset_path = 'dataset/fer2013plus/fer2013/test' 


# Load the FER dataset
fer_dataset = DatasetFolder(root=fer_dataset_path, loader=Image.open, extensions=".png", transform=transform)
print(f"Number of images in the FER dataset: {len(fer_dataset)}")


# Create DataLoader for testing
batch_size = 1 # Adjust batch size based on available memory
fer_loader = DataLoader(fer_dataset, batch_size=batch_size, shuffle=False)

total_correct = 0
total_samples = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with torch.no_grad():
    for images, labels in fer_loader:
        images = images.to(device)
        labels = labels.to(device)

        predicted = model.predict_emotion_fer(images)
        
        EMOTIONS_MAPPING = {
        "Angry": 0,
        "Disgust": 1,
        "Fear": 2,
        "Happy": 3,
        "Sad": 4,
        "Surprise": 5,
        "Neutral": 6,
        "Contempt": 7,
        }
        
        REVERSE_EMOTIONS_MAPPING = {v: k for k, v in EMOTIONS_MAPPING.items()}
        true_labels = REVERSE_EMOTIONS_MAPPING[labels.item()]
                        
        print(f"Predicted: {predicted} vs Label: {true_labels}")

        total_samples += 1
        
        total_correct += (predicted == true_labels)

        if(total_samples%1000 == 0):
            print(f"Samples so far: {total_samples}, Accuracy so far: {(total_correct/total_samples):.2f}%")


test_accuracy = (total_correct / total_samples) * 100
print(f"Test Accuracy on FER Dataset: {test_accuracy:.2f}%")