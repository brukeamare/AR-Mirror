import cv2
import mediapipe as mp
from PyQt5.QtWidgets import QApplication, QWidget, QLabel
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer

import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh


class CameraApp(QWidget):
    def __init__(self, overlay_path='overlay2.png', overlay_type='eye'):
        super().__init__()
        self.initUI()
        
        self.update_overlay(overlay_path, overlay_type)

        # Load your overlay image here
        self.overlay_img = cv2.imread('overlay2.png', -1)  # Ensure this is the correct path

        # Initialize mediapipe face detection
        self.mp = mp.solutions


        # Frame skipping variables
        self.skip_frames = 5  # Number of frames to skip
        self.frame_count = 0  # Current frame count
        
        self.face_mesh = self.mp.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def initUI(self):
        self.setWindowTitle("Camera Feed with Filter")
        self.setGeometry(100, 100, 800, 600)
        self.image_label = QLabel(self)
        self.image_label.resize(800, 600)

        # Timer for updating the feed
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(20)  # Adjust the refresh rate as needed

        # Start the camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open video capture device.")
            return
        
    def update_overlay(self, overlay_path, overlay_type):
        """Update the overlay image and its type."""
        self.overlay_img = cv2.imread(overlay_path, -1)  # Load the new overlay image
        if self.overlay_img is None:
            raise ValueError(f"Could not load overlay image from {overlay_path}")
        self.overlay_type = overlay_type  


    def update_frame(self):
        ret, frame = self.cap.read()
        self.frame_count += 1

        if ret and self.frame_count % self.skip_frames == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(frame_rgb)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Example: Calculate bounding box of the face
                    h, w, _ = frame.shape
                    landmark_array = [(landmark.x * w, landmark.y * h) for landmark in face_landmarks.landmark]

                    # Depending on the overlay type, calculate the size and position differently
                    if self.overlay_type == 'face':
                        # Use full face for sizing and positioning
                        x_min, y_min = min(landmark_array, key=lambda p: p[0])[0], min(landmark_array, key=lambda p: p[1])[1]
                        x_max, y_max = max(landmark_array, key=lambda p: p[0])[0], max(landmark_array, key=lambda p: p[1])[1]
                    elif self.overlay_type == 'eye':
                        # Define eye landmarks indices
                        right_eye_indices = [130, 133]  # Just as an example, check the exact indices from the documentation
                        left_eye_indices = [359, 362]  # Just as an example, check the exact indices from the documentation

                        # Calculate bounding box for the eyes
                        eye_landmarks = [face_landmarks.landmark[idx] for idx in right_eye_indices + left_eye_indices]
                        eye_landmarks_array = [(landmark.x * w, landmark.y * h) for landmark in eye_landmarks]

                        x_min = min(eye_landmarks_array, key=lambda p: p[0])[0]
                        y_min = min(eye_landmarks_array, key=lambda p: p[1])[1]
                        x_max = max(eye_landmarks_array, key=lambda p: p[0])[0]
                        y_max = max(eye_landmarks_array, key=lambda p: p[1])[1]

                    elif self.overlay_type == 'head':
                        # Define head landmark indices (extreme top, bottom, left, and right points)
                        head_top_index = 10  # Example, top of the forehead
                        chin_bottom_index = 152  # Example, bottom of the chin
                        left_face_index = 234  # Example, left side of the face
                        right_face_index = 454  # Example, right side of the face

                        # Calculate bounding box for the head
                        head_landmarks = [face_landmarks.landmark[idx] for idx in [head_top_index, chin_bottom_index, left_face_index, right_face_index]]
                        head_landmarks_array = [(landmark.x * w, landmark.y * h) for landmark in head_landmarks]

                        x_min = min(head_landmarks_array, key=lambda p: p[0])[0]
                        y_min = min(head_landmarks_array, key=lambda p: p[1])[1]
                        x_max = max(head_landmarks_array, key=lambda p: p[0])[0]
                        y_max = max(head_landmarks_array, key=lambda p: p[1])[1]


                    # Calculate overlay size and position
                    overlay_width = x_max - x_min
                    overlay_height = y_max - y_min
                    overlay_aspect_ratio = self.overlay_img.shape[1] / self.overlay_img.shape[0]
                    scale_factor = overlay_width / self.overlay_img.shape[1]
                    new_overlay_width = int(overlay_width)
                    new_overlay_height = int(new_overlay_width / overlay_aspect_ratio)

                    # Resize overlay image proportionally
                    overlay_resized = cv2.resize(self.overlay_img, (new_overlay_width, new_overlay_height))

                    # Calculate position for the overlay
                    overlay_x_start = int(x_min)
                    overlay_y_start = max(int(y_min - new_overlay_height / 2), 0) if self.overlay_type == 'head' else int(y_min)

                    # Overlay the image onto the frame
                    for i in range(overlay_resized.shape[0]):
                        for j in range(overlay_resized.shape[1]):
                            if overlay_resized[i, j][3] != 0:  # Check for transparency
                                frame[overlay_y_start + i, overlay_x_start + j] = overlay_resized[i, j][:3]

            # Convert the frame back to Qt format to display in the GUI
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            convert_to_Qt_format = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            p = convert_to_Qt_format.scaled(self.image_label.width(), self.image_label.height())
            self.image_label.setPixmap(QPixmap.fromImage(p))



    def closeEvent(self, event):
        self.cap.release()

if __name__ == '__main__':
    app = QApplication([])
    ex = CameraApp()
    ex.update_overlay('overlay2.png', 'eye') 
    ex.show()
    app.exec_()
