import pandas as pd
import os
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import efficientnet_b4,EfficientNet_B4_Weights
from torchvision.models import efficientnet_b0,EfficientNet_B0_Weights
from torchvision.models import resnet50
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import DatasetFolder
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFilter
import torchvision
from torch.optim import lr_scheduler
from torchvision.transforms.functional import erase
import matplotlib.pyplot as plt
# Import the re module for regular expressions
import re

class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        image_path = self.dataframe.loc[idx, "image_path"]
        image = Image.open(image_path)  # Use torchvision.io to read the image
        label = int(self.dataframe.loc[idx, "label"])
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Define a function to filter parameters
def filter_params(params, include_patterns, exclude_patterns):
    included_params = []
    excluded_params = []
    for name, param in params:
        if any(re.search(pattern, name) for pattern in include_patterns):
            included_params.append(param)
        elif not any(re.search(pattern, name) for pattern in exclude_patterns):
            excluded_params.append(param)
    return included_params, excluded_params

# Load the annotations for training and validation from separate CSV files
train_annotations_path = 'dataset/AffectNet/affect_train.csv'
valid_annotations_path = 'dataset/AffectNet/affect_val.csv'
train_annotations_df = pd.read_csv(train_annotations_path)
valid_annotations_df = pd.read_csv(valid_annotations_path)


label_mapping = {
    'Anger': 0,
    'Disgust': 1,
    'Fear': 2,
    'Happiness': 3,
    'Sadness': 4,
    'Surprise': 5,
    'Neutral': 6,
    'Contempt' :7,
}

# Apply the mapping to the labels in the dataframes
train_annotations_df['label'] = train_annotations_df['label'].map(label_mapping)
valid_annotations_df['label'] = valid_annotations_df['label'].map(label_mapping)
num_classes = 8



transform = transforms.Compose([
    # # transforms.RandomHorizontalFlip(), #Mal testen, ist das sinnvoll?
    # transforms.RandomResizedCrop(size=190, scale=(0.8, 1.0)),
    # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),

    # transforms.RandomPerspective(),
    transforms.ElasticTransform(alpha=5.0, sigma=5.0),
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),

    # transforms.RandomApply([transforms.Lambda(lambda x: elastic_transform(x, alpha_range=(50, 100), sigma_range=(5, 15)))]),
    # transforms.Lambda(lambda x: erase(x, i=0, j=0, h=x.size(1), w=x.size(2), v=0)),  # Apply cutout to the entire image

    transforms.RandomGrayscale(p=0.25),
    transforms.RandomRotation(degrees=15),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(0.15, 0.15, 0.15),
    # transforms.RandAugment(),
    torchvision.transforms.RandomAutocontrast(p=0.4),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

#First Model: transforms.RandomRotation(degrees=15),
    # transforms.RandomVerticalFlip(),
    # transforms.ColorJitter(0.15, 0.15, 0.15),
    # transforms.RandAugment(),
    # torchvision.transforms.RandomAutocontrast(p=0.5),
    # transforms.ToTensor(),
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

transform_valid =transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# Create dataset and data loaders
train_dataset = CustomDataset(dataframe=train_annotations_df, transform=transform)
valid_dataset = CustomDataset(dataframe=valid_annotations_df, transform=transform_valid)

batch_size = 64 #
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=24)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,num_workers=24)

# Load the EfficientNetB0 model
# base_model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)  # Load EfficientNet-B4
base_model = efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)  # Load EfficientNet-B4
num_features = base_model.classifier[1].in_features
base_model.classifier[1] = nn.Linear(in_features=num_features, out_features=num_classes)  # Adjust classifier layer
model = base_model

# base_model = models.inception_v3(pretrained=True)
# num_features = base_model.fc.in_features  # InceptionV3 uses fc layer, InceptionV4 uses last_linear layer

# # Replace the fully connected layer with a new one
# base_model.fc = nn.Linear(in_features=num_features, out_features=num_classes)
# model = base_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# model_c = torch.compile(model) #Not possible with inception v3
model_c = model
# Define loss function and optimizer
criterion = nn.CrossEntropyLoss() #First Model

# #New Idea: See Savchenko et al (2022): Weighted cross entropy loss
# class_counts = train_annotations_df['label'].value_counts().sort_index()
# class_weights = 1.0 / torch.Tensor(class_counts)
# criterion = nn.CrossEntropyLoss(weight=class_weights)

# Define patterns to include/exclude BatchNorm parameters
include_patterns = [r'^(?!.*\.bn)']  # Match any layer name that doesn't contain '.bn' 
exclude_patterns = [r'.*\.bn.*']

# Filter parameters for weight decay and no weight decay
params_to_decay, params_not_to_decay = filter_params(model.named_parameters(), include_patterns, exclude_patterns)

# optimizer = optim.AdamW(model.parameters(), lr=3e-04, weight_decay=0.2) 
# optimizer = optim.AdamW(model.parameters(), lr=3e-05, weight_decay=0.2)  #FirstModel

# Create optimizer with different weight decay for different parameter groups
optimizer = optim.AdamW([
    {'params': params_to_decay, 'weight_decay': 0.2},  # Apply weight decay to these parameters
    {'params': params_not_to_decay, 'weight_decay': 0.0}  # Exclude weight decay for these parameters
], lr=4e-05)



epochs = 50 

#Learning rate Scheduler
lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max = batch_size*epochs)

# Train the model


scaler = torch.cuda.amp.GradScaler()

best_valid_loss= 100

for epoch in range(epochs):
    model_c.train()
    # Adjust learning rate using the scheduler
    
    total_train_correct = 0
    total_train_samples = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            output = model_c(images)
            loss = criterion(output.cuda(), labels.cuda())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

        _, train_predicted = torch.max(output, 1)
        total_train_samples += labels.size(0)
        total_train_correct += (train_predicted == labels).sum().item()
        
    train_accuracy = (total_train_correct / total_train_samples) * 100
    
    model_c.eval()
    valid_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model_c(images)
            loss = criterion(outputs.cuda(), labels.cuda())
            valid_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f"Epoch [{epoch+1}/{epochs}] - "
          f"Validation Loss: {valid_loss/len(valid_loader):.4f}, "
          f"Validation Accuracy: {(correct/total)*100:.2f}%"
          f", Training Accuracy: {train_accuracy:.2f}%, ")
    #TBD: Valid loss überschreiben, dann model speichern wie unten, wenn kleiner als zuvor

    if(valid_loss < best_valid_loss):
        best_valid_loss = valid_loss
        #mit accuracy abspeichern im namen mglich
        torch.save(model.state_dict(), 'affectNet_emotion_model_best.pth')

# Save the trained model
torch.save(model.state_dict(), 'affectNet_emotion_model.pth')

