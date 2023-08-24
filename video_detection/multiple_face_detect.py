import time
from keras.models import model_from_json
import numpy as np
import cv2
import argparse
import os
from models.new_model import affect_model

from models.new_model.affect_model import AffectModel


from models.facial_expression_model import FacialExpressionModel


# Variables for video caption and source these can be defined later on when starting the file
# THIS MIGHT BE DIFFERENT FOR YOU BC I AM ON MACOS! SO BE AWARE THIS MIGHT BE A SOURCE OF ERROR WHEN U TRY TO START UP
# Changed fps to an optional argument --> Now you can just pass "Default" as the source argument and no fps limit is set and source 0 (to avoid start up problems on windows)

parser = argparse.ArgumentParser()
parser.add_argument("--source", default=0)
parser.add_argument("--fps", nargs="?", default=30)
args = parser.parse_args()


if(args.source == "Default"):
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture(
        # os.path.abspath(args.source) if not args.source == "webcam" else 1
        0
    )
    cap.set(cv2.CAP_PROP_FPS, int(args.fps))

# load the facecascade model
faceCascade = cv2.CascadeClassifier("./models/haarcascade_frontalface_default.xml")

font = cv2.FONT_HERSHEY_SIMPLEX

# define the time for a new face to be considered valid
VALID_FACE_TIME = 3

# define the time to keep the face id if no face is detected
FACE_ID_EXPIRY = 10

# maintain timers and face IDs for each area
face_timers = [None, None, None]
expiry_timers = [None, None, None]
face_ids = [None, None, None]
last_face_id = 0

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
    # gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(fr,cv2.COLOR_BGR2RGB)#
    
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)
    detected_faces = [DetectedFace(x, y, w, h) for x, y, w, h in faces]
    return detected_faces, fr, gray


# starting app and running face detection using OpenCV

def start_app(cnn):
    global last_face_id
    while cap.isOpened():
        faces, fr, gray_fr = getdata()
        
        # Sort detected faces by x-coordinate
        faces.sort(key=lambda face: face.x)

        height, width = fr.shape[:2] #Height and width of frame

        #Thresholds mark vertical lines --> when the middle of a recognized face crosses the line the index will be changed
        threshold_index1 = int(width/3) # so far just sliced into three same-sized areas --> has to be adapted to fit the use case
        threshold_index2 = int(2 * width/3)
        threshold_indices = [0, threshold_index1, threshold_index2]

        max_prob_faces = [None, None, None] # List to hold the faces with maximum probability in each area

        # split faces into the respective areas
        area1_faces = [face for face in faces if face.x + face.w/2 < threshold_index1]
        area2_faces = [face for face in faces if face.x + face.w/2 >= threshold_index1 and face.x + face.w/2 < threshold_index2]
        area3_faces = [face for face in faces if face.x + face.w/2 >= threshold_index2]

        area_faces = [area1_faces, area2_faces, area3_faces]   

        # Get the face with the highest probability from each area. This depends on the probability given by faceCascade.detectMultiScale
        for i in range(3):
            if not area_faces[i]:
                continue
            max_prob_face = max(area_faces[i], key=lambda x: x.w * x.h) # assuming w*h (area of face rectangle) as the proxy for face probability 
            max_prob_face.id = i + 1
            max_prob_faces[i] = max_prob_face

        
        #Lines are there to show the areas for the id's -> Can later be removed
        cv2.line(
            img = fr,
            pt1 = (threshold_index1,0),
            pt2 = (threshold_index1, height),
            color=(0, 0, 0),
            thickness=10
            )
        cv2.line(
            img = fr,
            pt1 = (threshold_index2, 0),
            pt2 = (threshold_index2, height),
            color=(0, 0, 0),
            thickness=10,
            )

        current_time = time.time()

        for i in range(3):
            # if a face was detected
            if max_prob_faces[i] is not None:
                expiry_timers[i] = None
                if face_ids[i] is None:
                    if face_timers[i] is None:
                        # if no ID has been assigned yet, start the timer
                        face_timers[i] = current_time + VALID_FACE_TIME
                        
                    # if an ID has been assigned and the face has been detected continuously for VALID_FACE_TIME seconds, confirm the ID
                    elif current_time >= face_timers[i]:
                        face_ids[i] = last_face_id + 1
                        last_face_id += 1
      
            else:
                if expiry_timers[i] is None and face_timers[i] is not None:
                    expiry_timers[i] = current_time + FACE_ID_EXPIRY
                
                # if no face was detected, reset the timer if it has been more than FACE_ID_EXPIRY seconds
                elif expiry_timers[i] is not None and current_time >= expiry_timers[i]:
                    face_ids[i] = None
                    face_timers[i] = None
                    expiry_timers[i] = None
                

        # display the timers in the video output
        for i in range(3):
            threshold_index = threshold_indices[i]  
            if expiry_timers[i] is not None:
                cv2.putText(fr, f"Expiry: {int(expiry_timers[i] - current_time)}", (threshold_index, 50), fontFace =font, fontScale = 1, color = (0, 0, 0), thickness=2)
            elif face_timers[i] is not None:
                cv2.putText(fr, f"Timer: {int(current_time - face_timers[i])}", (threshold_index, 50), fontFace = font, fontScale = 1, color = (0, 0, 0), thickness=2)
        
        # Predict emotions and draw bounding boxes
        for i in range(3):
            face = max_prob_faces[i]
            if face is None:
                continue
            fc = gray_fr[face.y : face.y + face.h, face.x : face.x + face.w]

            roi = cv2.resize(fc, (214, 214))
            pred = cnn.predict_emotion(roi)

            emotioncolor = AffectModel.EMOTIONS_COLOR_MAPPING[pred]

            # Adding Rectangle and the text that displays the detected emotion and ID
            cv2.putText(
                fr,
                # f"{pred} ({face.id})", 
                f"({face_ids[i]}) {pred}", # currently changed to highlight index changes
                (face.x, face.y),
                fontFace=font,
                fontScale=1,
                color=emotioncolor,
                thickness=1,
            )
            cv2.rectangle(
                fr,
                (face.x, face.y),
                (face.x + face.w, face.y + face.h),
                color = emotioncolor,
                thickness= 1,
            )

        # break loop if q is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        cv2.namedWindow("Facial Emotion Recognition", cv2.WINDOW_NORMAL)
        cv2.imshow("Facial Emotion Recognition", fr)

    cap.release()

    # closing window after stopping the application
    cv2.destroyAllWindows()




if __name__ == "__main__":
    # model = FacialExpressionModel("./models/model.json", "./models/weights.h5")
    model = AffectModel("affectNet_emotion_model_best_1.49.pth")
    start_app(model)