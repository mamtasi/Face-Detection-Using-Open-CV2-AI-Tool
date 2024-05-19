import cv2
import os
import numpy as np

# Set the paths and filenames
haar_file = 'haarcascade_frontalface_default.xml'
datasets = 'datasets'

# Initialize variables
(images, labels, names) = ([], [], {})
id = 0

# Read images from the dataset directory
for subdir, dirs, files in os.walk(datasets):
    for dir_name in dirs:
        names[id] = dir_name
        subject_path = os.path.join(datasets, dir_name)
        for file_name in os.listdir(subject_path):
            image_path = os.path.join(subject_path, file_name)
            label = id
            images.append(cv2.imread(image_path, 0))
            labels.append(int(label))
        id += 1

# Convert the image and label lists to NumPy arrays
(images, labels) = np.array(images), np.array(labels)

# Train the FisherFaceRecognizer model
model = cv2.face.FisherFaceRecognizer_create()
model.train(images, labels)

# Create the Haar cascade classifier
face_cascade = cv2.CascadeClassifier(haar_file)

def recognition():
    webcam = cv2.VideoCapture(0)
    while True:
        ret, frame = webcam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            face_resize = cv2.resize(face, (130, 100))
            prediction = model.predict(face_resize)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            if prediction[1] < 500:
                name = names[prediction[0]]
                cv2.putText(frame, name, (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
            else:
                name = "Unknown"
                cv2.putText(frame, name, (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
        cv2.imshow('Face Recognition', frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    webcam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognition()
