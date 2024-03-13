import cv2
import dlib
import numpy as np
from imutils import face_utils
from scipy.spatial import distance as dist

# Initialize dlib's face detector and the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Ensure this file is in your working directory

def eye_aspect_ratio(eye):
    # Compute the euclidean distances between the two sets of vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # Compute the euclidean distance between the horizontal eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

# Start video capture from the laptop's built-in camera
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break  # Break the loop if no frame is captured

    # Convert the captured frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    rects = detector(gray, 0)

    # Loop over the face detections
    for rect in rects:
        # Determine the facial landmarks for the face region
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # Extract the left and right eye coordinates
        leftEye = shape[42:48]
        rightEye = shape[36:42]

        # Calculate the EAR for both eyes
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # Average the EAR for both eyes for a general fatigue metric
        ear = (leftEAR + rightEAR) / 2.0

        # Display the EAR on the frame to indicate the level of tiredness
        cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display the resulting frame with fatigue analysis
    cv2.imshow('Fatigue Analysis', frame)
    
    # Check if the window was closed
    if cv2.getWindowProperty('Fatigue Analysis', cv2.WND_PROP_VISIBLE) < 1:        
        break

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
