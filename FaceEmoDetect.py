import cv2
from tensorflow.keras.models import load_model
import numpy as np

face_cascade_file = './models/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(face_cascade_file)

if face_cascade.empty():
    print("Error: Unable to load cascade classifier")
    exit()
    
emotion_model_file = './models/emotion_model.hdf5'
try:
    emotion_model = load_model(emotion_model_file) 
except Exception as e:
    print("Error: Unable to load emotion classification model:", e)
    exit()

emotion_labels = ['Angry','Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

def predict_emotion(face_roi):
    face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    face_roi = cv2.resize(face_roi, (64, 64)) 
    face_roi = np.expand_dims(face_roi, axis=0)
    face_roi = np.expand_dims(face_roi, axis=3)
    prediction = emotion_model.predict(face_roi)
    emotion_label = emotion_labels[np.argmax(prediction)]
    return emotion_label

image_path = r"C:\Users\cheru\OneDrive\Desktop\ian.jpg"
image = cv2.imread(image_path)

if image is None:
    print("Error: Unable to load image")
    exit()

faces = detect_faces(image)

for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    face_roi = image[y:y+h, x:x+w]
    emotion_label = predict_emotion(face_roi)
    cv2.putText(image, emotion_label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

cv2.imshow('Emotion Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
