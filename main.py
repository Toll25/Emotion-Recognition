import cv2
import numpy as np
from fer import FER
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

# Define the list to store multiple rectangle coordinates
rectangle_list = []
emotions_list = []


# Function to draw rectangles on the image
def draw_reference_id(image, rectangle, emotion, id):
    if rectangle:
        x, y, _, _ = rectangle
        reference_id_text = f"Ref ID: {id}"  # You can replace 123 with your reference ID
        cv2.putText(image, reference_id_text, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(image, emotion, (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


def draw_rectangles_and_emotions(image, rectangles, emotions):
    id=0
    for rect, emotion_dict in zip(rectangles, emotions):
        x, y, width, height = rect
        cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)

        # Sort emotions by score in descending order
        sorted_emotions = sorted(emotion_dict.items(), key=lambda x: x[1], reverse=True)

        # Display emotions near the rectangle in a separate window
        emotions_window = np.zeros((len(sorted_emotions) * 30, 200, 3), dtype=np.uint8)
        for i, (emotion, score) in enumerate(sorted_emotions):
            # Format the score as a percentage
            score_percentage = score * 100
            text = f"{emotion.upper()}: {score_percentage:.0f}%"
            cv2.putText(emotions_window, text, (10, i * 30 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Check if this emotion is the highest
            if i == 0:
                highest_emotion = emotion
                # Draw reference ID on the main frame with the highest emotion
                draw_reference_id(image, rect, highest_emotion, id)

        cv2.imshow("Emotions", emotions_window)


# Define the function that will run in a separate thread
def emotion_thread():
    global current_img
    global rectangle_list
    global emotions_list
    while True:
        # Acquire the lock to safely access the current_img variable
        with lock:
            img_copy = current_img.copy() if current_img is not None else None

        if img_copy is not None:
            detector = FER(mtcnn=True)
            result = detector.detect_emotions(img_copy)

            # Clear the existing rectangle list
            rectangle_list = []
            emotions_list = []

            if result:
                for results in result:
                    if "box" in results and len(results["box"]) == 4:
                        x, y, width, height = results["box"]
                        rectangle_list.append((x, y, width, height))
                        emotions_list.append(results["emotions"])


# Start the new thread
emotion_thread = threading.Thread(target=emotion_thread)

# Start the emotion recognition thread
emotion_thread.start()

while True:
    success, img = cam.read()

    # Acquire the lock to safely update the current_img variable
    with lock:
        current_img = img.copy()

    image_with_rectangles = img.copy()

    # Draw rectangles and display emotions on the image if there is at least one entry
    if rectangle_list:
        draw_rectangles_and_emotions(image_with_rectangles, rectangle_list, emotions_list)

    cv2.imshow("Webcam", image_with_rectangles)
    if cv2.waitKey(10) & 0xFF == 27:
        break

# Release the camera and close all windows
cam.release()
cv2.destroyAllWindows()
