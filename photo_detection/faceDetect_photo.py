from keras.models import model_from_json
import cv2
import numpy as np
from keras.preprocessing import image
from PIL import Image
import base64
from models.facial_expression_model import FacialExpressionModel


def is_smiling(picture):
    # convert the base64 object to jpg object and saving it to be called later on
    with open("./photo_detection/potential_smiler.jpg", "wb") as photo:
        photo.write(base64.b64decode(picture))

    # Load the cascades
    face_cascade = cv2.CascadeClassifier("./models/haarcascade_frontalface_default.xml")

    model = FacialExpressionModel("./models/model.json", "./models/weights.h5")

    # Open the image using PIL
    img = Image.open("./photo_detection/potential_smiler.jpg")

    # Convert image to numpy array
    img = np.array(img)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Iterate over the faces
    for x, y, w, h in faces:
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
