from matplotlib import transforms
import torch
from models.new_model.affect_model import AffectModel
from PIL import Image
import torch
from torchvision import transforms
from torchvision.datasets import DatasetFolder
from torch.utils.data import DataLoader
from models.facial_expression_model import FacialExpressionModel


# Model performance on the fer plus dataset is measured 
# Since the pictures of the fer plus dataset are in shape (48,48) and gray they have to be resized and converted to RGB in order to be processable for our model

#Load the trained model
model = AffectModel("affectNet_emotion_model_best_1.49.pth")


# Define the transformation -> Transform to RGB and resize to (224,224) [Input size of the trained efficientnet]
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Path to the FER dataset
fer_dataset_path = 'dataset/fer2013plus/fer2013/test' 

# Load the FER dataset
fer_dataset = DatasetFolder(root=fer_dataset_path, loader=Image.open, extensions=".png", transform=transform)
print(f"Number of images in the FER dataset: {len(fer_dataset)}")


# Create DataLoader for testing
batch_size = 1 # Adjust batch size based on available memory --> to easy the process we just use a batch size of 1 
fer_loader = DataLoader(fer_dataset, batch_size=batch_size, shuffle=False)

#Using cuda when its available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#The predictions have to be mapped to string labels to be com
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

total_samples = 0
category_correct = {emotion: 0 for emotion in EMOTIONS_MAPPING.keys()}
category_total = {emotion: 0 for emotion in EMOTIONS_MAPPING.keys()}

with torch.no_grad():
    for images, labels in fer_loader:
        images = images.to(device)
        labels = labels.to(device)

        predicted = model.predict_emotion_fer(images)
        
        
        true_labels = REVERSE_EMOTIONS_MAPPING[labels.item()]
                        
        print(f"Predicted: {predicted} vs Label: {true_labels}")

        total_samples += 1
        
        if predicted == true_labels:
            category_correct[true_labels] += 1
        category_total[true_labels] += 1

        if total_samples % 1000 == 0:
            print(f"Samples so far: {total_samples}, Total Accuracy so far: {(sum(category_correct.values()) / total_samples):.2f}%")

# Calculate accuracy for each emotion category
category_accuracies = {emotion: (category_correct[emotion] / category_total[emotion]) * 100 for emotion in EMOTIONS_MAPPING.keys()}

# Print accuracy for each emotion category
for emotion, accuracy in category_accuracies.items():
    print(f"Accuracy for {emotion}: {accuracy:.2f}%")

# Calculate total test accuracy
total_test_accuracy = (sum(category_correct.values()) / total_samples) * 100
print(f"Total Test Accuracy on FER Dataset: {total_test_accuracy:.2f}%")

