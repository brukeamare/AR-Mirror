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

import numpy as np

def apply_overlay_on_eyes(image, overlay_img, landmarks):
    # Indices for key points around the eyes in MediaPipe's 468 face landmarks
    LEFT_EYE_INDICES = [33, 133, 160, 158, 157, 173]
    RIGHT_EYE_INDICES = [362, 398, 384, 385, 386, 263]

    # Gather the eye landmarks
    left_eye_landmarks = [(int(landmarks[i].x * image.shape[1]), int(landmarks[i].y * image.shape[0])) for i in LEFT_EYE_INDICES]
    right_eye_landmarks = [(int(landmarks[i].x * image.shape[1]), int(landmarks[i].y * image.shape[0])) for i in RIGHT_EYE_INDICES]
    both_eyes_landmarks = left_eye_landmarks + right_eye_landmarks

    # Calculate the bounding box that covers both eyes
    x_coords, y_coords = zip(*both_eyes_landmarks)
    x_min, y_min = min(x_coords), min(y_coords)
    x_max, y_max = max(x_coords), max(y_coords)
    box_width, box_height = x_max - x_min, y_max - y_min

    # Determine the size of the overlay image to maintain aspect ratio
    overlay_aspect_ratio = overlay_img.shape[1] / overlay_img.shape[0]
    overlay_width = box_width
    overlay_height = int(overlay_width / overlay_aspect_ratio)

    # Adjust y_min to position the overlay centered vertically over the eyes
    y_min = y_min + (box_height - overlay_height) // 2

    # Resize the overlay image
    resized_overlay = cv2.resize(overlay_img, (overlay_width, overlay_height))

    # Overlay the image onto the frame, checking for transparency
    for i in range(overlay_height):
        for j in range(overlay_width):
            if resized_overlay[i, j][3] != 0:  # Check for transparency in the alpha channel
                y = y_min + i if (y_min + i) < image.shape[0] else image.shape[0] - 1
                x = x_min + j if (x_min + j) < image.shape[1] else image.shape[1] - 1
                image[y, x] = resized_overlay[i, j][:3]


# Load the overlay image.
overlay_img = cv2.imread('overlay3.png', cv2.IMREAD_UNCHANGED)
if overlay_img is None:
    raise ValueError("Could not load overlay image.")

# Capture video from the webcam.
cap = cv2.VideoCapture(0)
# Assuming you have a loop where you process each frame from the webcam
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Correct the function name based on your actual function definition
            apply_overlay_on_eyes(frame, overlay_img, face_landmarks.landmark)

    cv2.imshow('Frame', frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()
