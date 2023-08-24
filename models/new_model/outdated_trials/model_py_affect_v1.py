import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from efficientnet_pytorch import EfficientNet
from PIL import Image



class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data_df.iloc[idx, 0])
        image = Image.open(img_name)
        label = self.data_df.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, label

train_annotations_path = 'dataset/AffectNet/affect_train.csv'
valid_annotations_path = 'dataset/AffectNet/affect_val.csv'
train_annotations_df = pd.read_csv(train_annotations_path)
valid_annotations_df = pd.read_csv(valid_annotations_path)


train_image_dir = ""
valid_image_dir = ""
image_size = (224, 224)
batch_size = 32
num_classes = 8  # Set the number of classes based on your problem

# Define data transformations
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create custom datasets and data loaders
train_dataset = CustomDataset(train_annotations_path, train_image_dir, transform)
valid_dataset = CustomDataset(valid_annotations_path, valid_image_dir, transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

class CustomEfficientNet(nn.Module):
    def __init__(self, num_classes):
        super(CustomEfficientNet, self).__init__()
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b0')
        num_features = self.efficientnet._fc.in_features
        self.efficientnet._fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.efficientnet(x)

model = CustomEfficientNet(num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

model.to(device)

num_epochs = 10  # Adjust as needed

for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        valid_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            valid_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Validation Loss: {valid_loss/len(valid_loader):.4f}, "
              f"Validation Accuracy: {(correct/total)*100:.2f}%")

