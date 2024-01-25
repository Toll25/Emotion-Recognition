import threading
import cv2
import numpy
import os
from deepface import DeepFace

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
data_list = []

# It helps in identifying the faces

size = 4
haar_file = 'haarcascade_frontalface_default.xml'
datasets = 'datasets'

# Part 1: Create fisherRecognizer
print('Recognizing Face Please Be in sufficient Lights...')

# Create a list of images and a list of corresponding names
(images, labels, names, id) = ([], [], {}, 0)
for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(datasets, subdir)
        for filename in os.listdir(subjectpath):
            path = subjectpath + '/' + filename
            label = id
            images.append(cv2.imread(path, 0))
            labels.append(int(label))
        id += 1
(width, height) = (130, 100)

# Create a Numpy array from the two lists above
(images, labels) = [numpy.array(lis) for lis in [images, labels]]

# OpenCV trains a model from the images
# NOTE FOR OpenCV2: remove '.face'
model = cv2.face.LBPHFaceRecognizer_create()
model.train(images, labels)

# Part 2: Use fisherRecognizer on camera stream
face_cascade = cv2.CascadeClassifier(haar_file)


def draw_rectangles_and_emotions(image, data, faces):
    for face, (x, y, w, h) in zip(data, faces):
        region = face.get("region")
        box_x = region.get("x")
        box_y = region.get("y")
        box_height = region.get("h")
        box_width = region.get("w")
        cv2.rectangle(image, (box_x, box_y), (box_x + box_width, box_y + box_height), (255, 0, 0), 2)
        gender = face.get("dominant_gender")
        emotion = face.get("dominant_emotion")
        name = "not Recognized"

        gender_percent = convertFloatToIntString(face.get("gender").get(gender))
        emotion_percent = convertFloatToIntString(face.get("emotion").get(emotion))

        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))
        # Try to recognize the face
        prediction = model.predict(face_resize)
        if prediction[1] < 95:
            name = '% s - %.0f' % (names[prediction[0]], prediction[1])

        # Draw reference ID on the main frame with the highest emotion
        cv2.putText(image, "Name: " + name, (x, y - 30), cv2.QT_FONT_NORMAL, 0.5, (255, 0, 0), 2)
        cv2.putText(image, "Emotion: " + emotion + " (" + emotion_percent + "%)",
                    (x, y - 50), cv2.QT_FONT_NORMAL, 0.5, (255, 0, 0), 2)
        cv2.putText(image, "Gender: " + gender + " (" + gender_percent + "%)", (x, y - 70), cv2.QT_FONT_NORMAL,
                    0.5, (255, 0, 0), 2)

        # Define the function that will run in a separate thread


def convertFloatToIntString(floatValue):
    return str(numpy.int8(numpy.rint(floatValue)))


def emotion_thread():
    global current_img
    global rectangle_list
    global data_list
    while True:
        # Acquire the lock to safely access the current_img variable
        with lock:
            img_copy = current_img.copy() if current_img is not None else None

        if img_copy is not None:
            cv2.imwrite("img.png", img_copy)
            deep_results = DeepFace.analyze(img_path="img.png",
                                            actions=['gender', 'emotion'],
                                            enforce_detection=False
                                            )

            # Clear the existing rectangle list
            data_list = []

            if deep_results:
                for result in deep_results:
                    if result.get("face_confidence") == 0:
                        print("No Face")
                    else:
                        data_list = deep_results


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

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    data_faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Draw rectangles and display emotions on the image if there is at least one entry
    if data_list:
        draw_rectangles_and_emotions(image_with_rectangles, data_list, data_faces)

    cv2.imshow("Webcam", image_with_rectangles)
    key = cv2.waitKey(10)
    if key == 27:
        break

# Release the camera and close all windows
cam.release()
cv2.destroyAllWindows()
