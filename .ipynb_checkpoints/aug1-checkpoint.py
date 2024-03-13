import cv2
import mediapipe as mp
from PyQt5.QtWidgets import QApplication, QWidget, QLabel
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer

class CameraApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

        # Load your overlay image here
        self.overlay_img = cv2.imread('overlay2.png', -1)  # Ensure this is the correct path

        # Initialize mediapipe face detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.5)

        # Frame skipping variables
        self.skip_frames = 5  # Number of frames to skip
        self.frame_count = 0  # Current frame count

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
        
        


    def update_frame(self):
        ret, frame = self.cap.read()
        self.frame_count += 1

        if ret and self.frame_count % self.skip_frames == 0:  # Process every nth frame
            # Convert the frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(frame_rgb)

            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape  # Frame shape
                    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                                 int(bboxC.width * iw), int(bboxC.height * ih)
                    
                    # Resize overlay image to fit the face
                    overlay_image = cv2.resize(self.overlay_img, (w, h))

                    # Overlay the image
                    for i in range(h):
                        for j in range(w):
                            if overlay_image[i, j][3] != 0:  # Check alpha channel
                                frame[y + i, x + j] = overlay_image[i, j][:3]

            # Convert the frame to QT format for display
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
    ex.show()
    app.exec_()
