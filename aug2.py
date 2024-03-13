import cv2
import mediapipe as mp

# Initialize MediaPipe Face Mesh.
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

import numpy as np  # Make sure to import NumPy

def apply_eye_overlay(image, overlay_img, landmarks):
    # Indices for the left and right eye in MediaPipe's 468 face landmarks.
    LEFT_EYE_INDICES = [33, 133, 160, 158, 157, 173]
    RIGHT_EYE_INDICES = [362, 398, 384, 385, 386, 263]

    # Calculate bounding box for the left and right eye.
    for eye_indices in [LEFT_EYE_INDICES, RIGHT_EYE_INDICES]:
        eye_landmarks = [(int(landmarks[i].x * image.shape[1]), int(landmarks[i].y * image.shape[0])) for i in eye_indices]
        
        # Use a NumPy array for the eye landmarks
        eye_landmarks_np = np.array(eye_landmarks, dtype=np.int32)

        # Calculate the bounding rectangle for the eye landmarks
        x, y, w, h = cv2.boundingRect(eye_landmarks_np.reshape(-1, 1, 2))

        # Resize overlay to eye bounding box size.
        resized_overlay = cv2.resize(overlay_img, (w, h))

        # Overlay the image onto the frame, checking boundaries.
        for i in range(h):
            for j in range(w):
                if resized_overlay[i, j][3] != 0:  # Check for transparency in the alpha channel.
                    overlay_x = x + j
                    overlay_y = y + i
                    if overlay_x < image.shape[1] and overlay_y < image.shape[0]:  # Check boundaries
                        image[overlay_y, overlay_x] = resized_overlay[i, j][:3]


# Load the overlay image.
overlay_img = cv2.imread('overlay3.png', cv2.IMREAD_UNCHANGED)
if overlay_img is None:
    raise ValueError("Could not load overlay image.")

# Capture video from the webcam.
cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Convert the frame to RGB.
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Face Mesh.
    results = face_mesh.process(frame_rgb)

    # Draw the face mesh annotations on the frame.
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            apply_eye_overlay(frame, overlay_img, face_landmarks.landmark)

    # Display the resulting frame.
    cv2.imshow('Overlay Application', frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
