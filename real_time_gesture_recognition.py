import cv2
import numpy as np
import mediapipe as mp
from keras.src.saving.saving_api import load_model

# Load the trained model
model = load_model("E:/VS Code/ProdiyInfoTech/hand_gesture_classifier.h5")

# List of gesture labels (must match the training order)
class_names = [
    'call_me', 'fingers_crossed', 'okay', 'paper', 'peace',
    'rock', 'rock_on', 'scissor', 'thumbs', 'up', 'other'
]

# Initialize MediaPipe Hand model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip and convert the image to RGB
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get bounding box around hand
            img_height, img_width, _ = frame.shape
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]

            xmin = int(min(x_coords) * img_width) - 20
            ymin = int(min(y_coords) * img_height) - 20
            xmax = int(max(x_coords) * img_width) + 20
            ymax = int(max(y_coords) * img_height) + 20

            # Clip to frame boundaries
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(img_width, xmax)
            ymax = min(img_height, ymax)

            hand_img = frame[ymin:ymax, xmin:xmax]
            if hand_img.size == 0:
                continue

            # Preprocess hand image
            gray_hand = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
            resized_hand = cv2.resize(gray_hand, (64, 64))
            normalized_hand = resized_hand / 255.0
            input_hand = np.expand_dims(normalized_hand, axis=(0, -1))  # Shape: (1, 64, 64, 1)

            # Predict gesture
            prediction = model.predict(input_hand)
            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction)
            label = f"{class_names[predicted_class]} ({confidence:.2f})"

            # Display result
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, label, (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Real-Time Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
