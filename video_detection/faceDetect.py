from keras.models import model_from_json
import numpy as np
import cv2
import argparse
import os

class FacialExpressionModel(object):
    # detectable Emotions
    EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

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
        return FacialExpressionModel.EMOTIONS_LIST[np.argmax(self.preds)]

# Variables for video caption and source these can be defined later on when starting the file 
# THIS MIGHT BE DIFFERENT FOR YOU BC I AM ON MACOS! SO BE AWARE THIS MIGHT BE A SOURCE OF ERROR WHEN U TRY TO START UP
parser = argparse.ArgumentParser()
parser.add_argument("source")
parser.add_argument("fps")
args = parser.parse_args()
cap = cv2.VideoCapture(os.path.abspath(args.source) if not args.source == 'webcam' else 1)
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX
cap.set(cv2.CAP_PROP_FPS, int(args.fps))

# Function for grayscaling the videoimage and using the haarcascade to detect faces
def getdata():
    _, fr = cap.read()
    gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)
    return faces, fr, gray

# starting app and running face detection using OpenCV
def start_app(cnn):
    while cap.isOpened():

        # There's two actual images I use here: 
        # first the grayscaled which is used for the haarcascade and 
        # second the actual image in color which is displayed to the user and contains not only boxes but also labelling of the emotions
        faces, fr, gray_fr = getdata()
        for (x, y, w, h) in faces:
            fc = gray_fr[y:y + h, x:x + w]
            roi = cv2.resize(fc, (48, 48))
            pred = cnn.predict_emotion(roi[np.newaxis, :, :, np.newaxis])

            # Adding Rectangle and the text that displays the detected emotion
            cv2.putText(fr, pred, (x, y), font, 3, (255, 255, 0), 1)
            cv2.rectangle(fr, (x, y), (x + w, y + h), (255, 0, 0), 2)
        if cv2.waitKey(1) == 27:
            break
        cv2.imshow('Facial Emotion Recognition', fr)
    cap.release()
    # closing window after stopping the application
    cv2.destroyAllWindows()


if __name__ == '__main__':
    model = FacialExpressionModel("model.json", "weights.h5")
    start_app(model)