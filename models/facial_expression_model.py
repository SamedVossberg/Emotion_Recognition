from keras.models import model_from_json
import numpy as np


class FacialExpressionModel(object):
    # detectable Emotions
    EMOTIONS_COLOR_MAPPING = {
        "Angry": (255, 0, 0),
        "Disgust": (0, 255, 255),
        "Fear": (255, 0, 255),
        "Happy": (0, 255, 0),
        "Sad": (0, 0, 0),
        "Surprise": (255, 255, 0),
        "Neutral": (255, 255, 255),
    }
    EMOTIONS_LIST = list(EMOTIONS_COLOR_MAPPING.keys())

    def __init__(self, model_json_file, model_weights_file):
        # loading of model and weights
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)
        self.loaded_model.load_weights(model_weights_file)
        print("Model loaded from disk")
        self.loaded_model.summary()

    # function for Emotion detection returning the FaceialExpressionModel emotions
    def predict_emotion(self, img):
        self.preds = self.loaded_model.predict(img)
        print(self.preds)
        return FacialExpressionModel.EMOTIONS_LIST[np.argmax(self.preds)]
