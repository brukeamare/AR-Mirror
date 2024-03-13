import cv2
import dlib
import numpy as np
from imutils import face_utils

# Initialize face and eye detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Ensure this file is in your working directory

def process_dark_circles(roi_gray):
    # Applying thresholding to highlight dark areas (potential dark circles)
    _, binary = cv2.threshold(roi_gray, 30, 255, cv2.THRESH_BINARY_INV)  # Threshold might need adjustment
    return binary

def process_lines_and_wrinkles(roi_gray):
    # Using edge detection to highlight lines and wrinkles
    edges = cv2.Canny(roi_gray, 50, 150)  # Parameters might need adjustment
    return edges

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        # Processing for each eye
        for eye_points in [range(36, 42), range(42, 48)]:  # Left eye, Right eye
            # Extracting the eye region
            (x, y, w, h) = cv2.boundingRect(np.array([landmarks[eye_points]]))
            roi_gray = gray[y:y+h, x:x+w]

            # Process dark circles and lines/wrinkles
            dark_circles_mask = process_dark_circles(roi_gray)
            lines_mask = process_lines_and_wrinkles(roi_gray)

            # Create a 3-channel overlay for visualization
            overlay = np.zeros((h, w, 3), dtype='uint8')
            overlay[dark_circles_mask > 0] = (0, 0, 255)  # Red overlay for dark circles
            overlay[lines_mask > 0] = (255, 0, 0)  # Blue overlay for lines/wrinkles

            # Overlay the processed regions back onto the frame
            frame[y:y+h, x:x+w][overlay.any(axis=2)] = frame[y:y+h, x:x+w][overlay.any(axis=2)] * 0.5 + overlay[overlay.any(axis=2)] * 0.5

    # Display the resulting frame with overlays
    cv2.imshow("Fatigue Analysis", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
