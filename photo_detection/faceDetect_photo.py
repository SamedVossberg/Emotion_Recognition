from keras.models import model_from_json
import cv2
import numpy as np
from keras.preprocessing import image
from PIL import Image
import base64


class FacialExpressionModel(object):
    EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

    def __init__(self, model_json_file, model_weights_file):
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        self.loaded_model.load_weights(model_weights_file)
        print("Model loaded from disk")
        self.loaded_model.summary()

    def predict_emotion(self, img):
        self.preds = self.loaded_model.predict(img)
        return FacialExpressionModel.EMOTIONS_LIST[np.argmax(self.preds)]


def is_smiling(picture):

    # convert the base64 object to jpg object and saving it to be called later on
    with open("./potential_smiler.jpg", "wb") as photo:
        photo.write(base64.b64decode(picture))

    # Load the cascades
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    model = FacialExpressionModel("model.json", "weights.h5")

    # Open the image using PIL
    img = Image.open("./potential_smiler.jpg")

    # Convert image to numpy array
    img = np.array(img)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Iterate over the faces
    for (x, y, w, h) in faces:
        roi_color = gray[y : y + h, x : x + w]
        roi_color = cv2.resize(roi_color, (48, 48))
        # roi_color = image.img_to_array(roi_color)
        # roi_color = np.expand_dims(roi_color, axis=0)
        # roi_color = roi_color / 255.0
        predictions = model.predict_emotion(roi_color[np.newaxis, :, :, np.newaxis])
        print(predictions)
        if predictions == "Happy":
            return True

    # If no faces are smiling or no faces are detected, return False
    return False
