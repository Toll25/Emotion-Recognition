import time
import cv2
from fer import FER
import pprint
import threading

cam = cv2.VideoCapture(0)

cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # set width
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # set height
cam.set(cv2.CAP_PROP_BRIGHTNESS, 0)  # set brightness
cam.set(cv2.CAP_PROP_CONTRAST, 50)  # set contrast
cam.set(cv2.CAP_PROP_SATURATION, 64)  # call imshow() using plt object

# Create a lock to synchronize access to the current_img variable
lock = threading.Lock()
current_img = None

# Define the function that will run in a separate thread
def emotion_thread():
    global current_img
    while True:
        # Acquire the lock to safely access the current_img variable
        with lock:
            img_copy = current_img.copy() if current_img is not None else None

        if img_copy is not None:
            detector = FER()
            result = detector.detect_emotions(img_copy)
            pprint.pprint(result)
        time.sleep(1)

# Start the new thread
emotion_thread = threading.Thread(target=emotion_thread)

# Start the emotion recognition thread
emotion_thread.start()

while True:
    success, img = cam.read()

    # Acquire the lock to safely update the current_img variable
    with lock:
        current_img = img.copy()

    cv2.imshow("Webcam", img)
    if cv2.waitKey(10) & 0xFF == 27:
        break

# Release the camera and close all windows
cam.release()
cv2.destroyAllWindows()