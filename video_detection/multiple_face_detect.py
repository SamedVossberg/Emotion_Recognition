from keras.models import model_from_json
import numpy as np
import cv2
import argparse
import os

from models.facial_expression_model import FacialExpressionModel


# Variables for video caption and source these can be defined later on when starting the file
# THIS MIGHT BE DIFFERENT FOR YOU BC I AM ON MACOS! SO BE AWARE THIS MIGHT BE A SOURCE OF ERROR WHEN U TRY TO START UP
# parser = argparse.ArgumentParser()
# parser.add_argument("source")
# parser.add_argument("fps")
# args = parser.parse_args()
cap = cv2.VideoCapture(
    # os.path.abspath(args.source) if not args.source == "webcam" else 1
    0
)
faceCascade = cv2.CascadeClassifier("./models/haarcascade_frontalface_default.xml")
font = cv2.FONT_HERSHEY_SIMPLEX
# cap.set(cv2.CAP_PROP_FPS, int(args.fps))


class DetectedFace:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.id = None

    def get_center(self):
        return (self.x + self.w // 2, self.y + self.h // 2)


# Function for grayscaling the videoimage and using the haarcascade to detect faces
def getdata():
    _, fr = cap.read()
    gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)
    detected_faces = [DetectedFace(x, y, w, h) for x, y, w, h in faces]
    return detected_faces, fr, gray


# starting app and running face detection using OpenCV


def start_app(cnn):
    while cap.isOpened():
        faces, fr, gray_fr = getdata()
        num_faces = len(faces)
        if num_faces > 0:
            # Sort detected faces by x-coordinate
            faces.sort(key=lambda face: face.x)

            height, width = fr.shape[:2]
            threshold_index1 = int(width/3) # so far just sliced into three areas --> has to be adapted to fit the use case
            threshold_index2 = int(2 * width/3)

            for face in faces:
                if face.x + face.w/2 < threshold_index1:
                    face.id = 1
                elif face.x + face.w/2 < threshold_index2:
                    face.id = 2
                else:
                    face.id = 3

            # Predict emotions and draw bounding boxes
            for face in faces:
                fc = gray_fr[face.y : face.y + face.h, face.x : face.x + face.w]
                roi = cv2.resize(fc, (48, 48))
                pred = cnn.predict_emotion(roi[np.newaxis, :, :, np.newaxis])

                # Adding Rectangle and the text that displays the detected emotion and ID
                cv2.putText(
                    fr,
                    # f"{pred} ({face.id})",
                    f"{face.id}",
                    (face.x, face.y),
                    font,
                    3,
                    (255, 255, 0),
                    1,
                )
                cv2.rectangle(
                    fr,
                    (face.x, face.y),
                    (face.x + face.w, face.y + face.h),
                    (255, 0, 0),
                    2,
                )
                cv2.line(
                    img = fr,
                    pt1 = (threshold_index1,0),
                    pt2 = (threshold_index1, height),
                    color=(255, 0, 0)
                    )
                cv2.line(
                    img = fr,
                    pt1 = (threshold_index2, 0),
                    pt2 = (threshold_index2, height),
                    color=(255, 0, 0)
                    )
                

        if cv2.waitKey(1) == 27:
            break

        cv2.imshow("Facial Emotion Recognition", fr)

    cap.release()

    # closing window after stopping the application
    cv2.destroyAllWindows()


if __name__ == "__main__":
    model = FacialExpressionModel("./models/model.json", "./models/weights.h5")
    start_app(model)
