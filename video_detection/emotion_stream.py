import cv2
import numpy as np

from models.facial_expression_model import FacialExpressionModel


def video_smile_detector(file_path: str | int) -> None:
    # Load the cascade
    face_cascade = cv2.CascadeClassifier(
        "./photo_detection/haarcascade_frontalface_default.xml"
    )

    # Load the model
    model = FacialExpressionModel(
        "./photo_detection/model.json", "./photo_detection/weights.h5"
    )

    # Initialize video capture with the video file path or web cam
    cap = cv2.VideoCapture(file_path)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Break the loop if we've reached the end of the video
        if not ret:
            break

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Iterate over the faces
        for x, y, w, h in faces:
            roi_gray = gray[y : y + h, x : x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            roi_gray = roi_gray.astype("float") / 255.0
            roi_gray = np.expand_dims(roi_gray, axis=0)
            roi_gray = np.expand_dims(roi_gray, axis=-1)

            predictions = model.predict_emotion(roi_gray)
            print(predictions)

            # If model predicts "Angry", draw a rectangle and label on the face
            if predictions == "Angry":
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    predictions,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )

        # Get the original frame's width and height
        original_height, original_width = frame.shape[:2]

        # Calculate the aspect ratio
        aspect_ratio = original_width / original_height

        # Check if width is larger than height
        if original_width > original_height:
            new_width = 720
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = 720
            new_width = int(new_height * aspect_ratio)

        # Resize the frame
        frame = cv2.resize(frame, (new_width, new_height))

        # Display the resulting frame
        cv2.imshow("Video", frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the capture and destroy all windows when done
    cap.release()
    cv2.destroyAllWindows()

# Change argument to a number for live streaming your web cam.
# Change argument to a path of a video file to test on a video. 
video_smile_detector("./data/videos/smiling.mov")
