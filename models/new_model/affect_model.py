import torch
import torch.nn as nn
from torchvision.transforms import ToTensor
from efficientnet_pytorch import EfficientNet
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights

class AffectModel(object):
    EMOTIONS_COLOR_MAPPING = {
        "Angry": (255, 0, 0),
        "Disgust": (0, 255, 255),
        "Fear": (255, 0, 255),
        "Happy": (0, 255, 0),
        "Sad": (0, 0, 0),
        "Surprise": (255, 255, 0),
        "Neutral": (255, 255, 255),
        "Contempt": (100,100,100)
    }
    EMOTIONS_LIST = list(EMOTIONS_COLOR_MAPPING.keys())

    def __init__(self, model_path):
        # Load the model's state dictionary
        model_state_dict = torch.load(model_path, map_location=torch.device('cpu'))

        # Initialize your model
        base_model = efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)
        # base_model = efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT)
        num_features = base_model.classifier[1].in_features
        base_model.classifier[1] = nn.Linear(num_features, out_features=8)  # Remove the original fully connected layer
        self.loaded_model = base_model  # Store the model as an attribute

        # Load the state dictionary into the model
        self.loaded_model.load_state_dict(model_state_dict)

        # Set the model to evaluation mode
        self.loaded_model.eval()
        print("Model loaded from disk")

    def predict_emotion(self, img):
        # Preprocess the input image
        transform = ToTensor()
        img_tensor = transform(img).unsqueeze(0)

        # Perform inference
        with torch.no_grad():
            self.preds = self.loaded_model(img_tensor)
            print(self.preds)
        
        emotion_idx = torch.argmax(self.preds).item()
        print(emotion_idx)
        return AffectModel.EMOTIONS_LIST[emotion_idx]

    #for fer:
    def predict_emotion_fer(self, img):
        # Perform inference
        with torch.no_grad():
            self.preds = self.loaded_model(img)

          
        emotion_idx = torch.argmax(self.preds).item()
        # print(emotion_idx)
        return AffectModel.EMOTIONS_LIST[emotion_idx]
