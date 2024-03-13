import cv2
import dlib
import numpy as np
from imutils import face_utils

# Initialize face and eye detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initial scale and translation values
scale_x, scale_y = 1.0, 1.0  # Scale adjustments
translate_x, translate_y = 0, 0  # Translation adjustments


def adjust_roi(x, y, w, h, adjustment_factor=1):
    """Adjust the ROI to include the area below the eyes with scaling and translation."""
    new_x = int((x + translate_x) * scale_x)
    new_y = int((y + h * adjustment_factor + translate_y) * scale_y)
    new_w = int(w * scale_x)
    new_h = int(h * (1 + adjustment_factor) * scale_y)
    return (new_x, new_y, new_w, new_h)

#  old version
# def adjust_roi(x, y, w, h, adjustment_factor=1):
#     """Adjust the ROI to include the area below the eyes."""
#     return (x, int(y + h * adjustment_factor), w, int(h * (1 + adjustment_factor)))

def get_reference_intensity(gray, landmarks):
    # Define points around the forehead area, for example, points 19 to 24 in the dlib 68 point model
    forehead_points = landmarks[19:25]  # Adjust indices as necessary for your model
    (x, y, w, h) = cv2.boundingRect(forehead_points)
    forehead_roi = gray[y:y+h, x:x+w]
    # Calculate the average intensity of the forehead region
    reference_intensity = np.mean(forehead_roi)
    return reference_intensity

def process_dark_circles(roi_gray, reference_intensity):
    # Calculate the average intensity of the under-eye region
    under_eye_intensity = np.mean(roi_gray)
    
    # Determine the difference in intensity from the reference
    intensity_diff = reference_intensity - under_eye_intensity
    
    # Apply a threshold to identify significantly darker areas
    if intensity_diff > 20:  # Threshold for 'darkness' can be adjusted
        _, binary = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # Increase the intensity of the red shading based on the difference
        red_intensity = np.clip(intensity_diff, 0, 255)  # Ensure within byte range
        return binary, red_intensity
    else:
        return np.zeros_like(roi_gray), 0

def process_lines_and_wrinkles(roi_gray, reference_intensity):
    """Enhance lines and wrinkles with a blue overlay based on their prominence."""
    # Apply GaussianBlur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(roi_gray, (5, 5), 0)

    # Use Sobel operator to detect edges in both horizontal and vertical directions
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

    # Calculate the magnitude of gradients
    magnitude = np.sqrt(sobelx**2 + sobely**2)

    # Normalize magnitude to the range of 0-255
    norm_magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    # Use the reference intensity to adjust the threshold for edge detection
    _, lines_mask = cv2.threshold(norm_magnitude, reference_intensity - 20, 255, cv2.THRESH_BINARY)

    # Apply morphological dilation to thicken the lines
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(lines_mask, kernel, iterations=1)

    # Calculate the intensity of the blue overlay
    # More prominent lines (higher magnitude) will have a stronger blue intensity
    blue_intensity = cv2.mean(dilated)[0]  # Average intensity of the dilated edges
    blue_intensity = np.clip(blue_intensity, 0, 255)  # Ensure within byte range

    return dilated, blue_intensity



def find_highest_video_device_index():
    index = 0
    highest_valid_index = -1  # Start with an invalid index
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:
            cap.release()
            break  # Exit the loop when a device fails to open
        else:
            highest_valid_index = index  # Update the highest valid index
            cap.release()
            index += 1
    return highest_valid_index

# Find the highest available device index
highest_index = find_highest_video_device_index()
if highest_index != -1:
    print(f"Using the highest device index found: {highest_index}")
    cap = cv2.VideoCapture(highest_index)
else:
    print("No available video capture devices found.")
    # Handle the case when no devices are found (e.g., exit the program or use a default device)

    
    
# Create a fullscreen window before starting the loop
cv2.namedWindow("Fatigue Analysis", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Fatigue Analysis", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
 
    
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Create a black image with the same size as the original frame to serve as the new background
    black_background = np.ones_like(frame) * 255
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        # Calculate reference intensity from the forehead region
        reference_intensity = get_reference_intensity(gray, landmarks)

        # Process under-eye regions for dark circles and lines/wrinkles
        for eye_points in [range(36, 42), range(42, 48)]:
            (x, y, w, h) = cv2.boundingRect(np.array([landmarks[eye_points]]))
            (x, y, w, h) = adjust_roi(x, y, w, h)  # Adjust ROI to focus on under-eye
            roi_gray = gray[y:y+h, x:x+w]

            dark_circles_mask, red_intensity = process_dark_circles(roi_gray, reference_intensity)
            lines_mask, blue_intensity = process_lines_and_wrinkles(roi_gray, reference_intensity)

            # Create a 3-channel overlay for visualization
            overlay = np.zeros((h, w, 3), dtype='uint8')
            overlay[dark_circles_mask > 0] = (0, 0, red_intensity)  # Red overlay for dark circles based on intensity
            overlay[lines_mask > 0] = (0, 165, 255)  # Apply orange overlay for lines/wrinkles

            # Apply the overlays to the black background instead of blending them with the original frame
            black_background[y:y+h, x:x+w][overlay.any(axis=2)] = overlay[overlay.any(axis=2)]

    cv2.imshow("Fatigue Analysis", black_background)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print('yeat')
        break
    elif cv2.waitKey(1) & 0xFF == ord('a'):
        translate_x -= 1  # Move overlay left
        print('translate_x =' +str(translate_x)+'')
    elif cv2.waitKey(1) & 0xFF == ord('d'):
        translate_x += 1  # Move overlay right
        print('translate_x =' +str(translate_x)+'')
    elif cv2.waitKey(1) & 0xFF == ord('w'):
        translate_y -= 1  # Move overlay up
        print('translate_y =' +str(translate_y)+'')
    elif cv2.waitKey(1) & 0xFF == ord('s'):
        translate_y += 1  # Move overlay down
        print('translate_y =' +str(translate_y)+'')
    elif cv2.waitKey(1) & 0xFF == ord('q'):
        scale_x -= 0.01  # Decrease horizontal scale
        print('scale_x =' +str(scale_x)+'')
    elif cv2.waitKey(1) & 0xFF == ord('e'):
        scale_x += 0.01  # Increase horizontal scale
        print('scale_x =' +str(scale_x)+'')
    elif cv2.waitKey(1) & 0xFF == ord('r'):
        scale_y -= 0.01  # Decrease vertical scale
        print('scale_y =' +str(scale_y)+'')
    elif cv2.waitKey(1) & 0xFF == ord('f'):
        scale_y += 0.01  # Increase vertical scale
        print('scale_y =' +str(scale_y)+'')


cap.release()
cv2.destroyAllWindows()
