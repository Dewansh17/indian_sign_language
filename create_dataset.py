import os
import pickle
import mediapipe as mp
import cv2
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []

# Define the length of the feature vector for 2 hands (each hand has 21 landmarks, each with x and y coordinates)
FEATURE_VECTOR_LENGTH = 42  # 21 landmarks * 2 (x and y) = 42 features for one hand
MAX_HANDS = 2  # We are considering a max of 2 hands per gesture

for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []

        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)

        # Process multiple hands if detected
        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                if hand_idx >= MAX_HANDS:
                    break  # Only process the first two hands

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            # Pad with zeros if there is only one hand
            if len(results.multi_hand_landmarks) < MAX_HANDS:
                padding = [0] * FEATURE_VECTOR_LENGTH  # Padding for one hand
                data_aux.extend(padding)

            # Ensure each sample has the same number of features
            data.append(data_aux)
            labels.append(dir_)

f = open('newdata70words.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()