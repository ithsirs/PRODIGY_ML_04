print("hello world")
import tensorflow as tf
from tensorflow import keras
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.src.saving.saving_api import load_model

# Load the model
model = load_model("gesture_classifier.h5")
class_names = ['Palm', 'Fist', 'L', 'Okay', 'Peace', 'Thumbs Up', 'Thumbs Down', 'Index Point', 'Pinch', 'Other']

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ROI: optional - crop to center or hand area
    roi = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi = cv2.resize(roi, (64, 64))
    roi = roi / 255.0
    roi = np.expand_dims(roi, axis=(0, -1))  # Shape: (1, 64, 64, 1)

    # Predict
    prediction = model.predict(roi)
    predicted_class = np.argmax(prediction)
    label = class_names[predicted_class]

    # Show result
    cv2.putText(frame, f"Prediction: {label}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

