import os
import pickle

import mediapipe as mp
import cv2
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Initialize Mediapipe Hands solution
mp_hands = mp.solutions.hands  # Load Mediapipe Hands module
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)  # Set up hand detection model

# Define dataset directory and expected number of features per sample
DATA_DIR = './data'  # Directory where dataset images are stored
EXPECTED_FEATURES = 42  # Expected number of features (21 hand landmarks × 2 coordinates)

# Lists to store processed data and corresponding labels
data = []
labels = []

# Loop through each class directory in the dataset
for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []  # Temporary list to store landmark features for one image
        x_ = []  # List to store x-coordinates of landmarks
        y_ = []  # List to store y-coordinates of landmarks

        # Read the image
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))  # Load the image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB (required by Mediapipe)

        # Process the image using Mediapipe Hands to extract hand landmarks
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:  # Check if any hands were detected
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x  # Extract x-coordinate
                    y = hand_landmarks.landmark[i].y  # Extract y-coordinate

                    x_.append(x)  # Store x-coordinate
                    y_.append(y)  # Store y-coordinate

                # Normalize landmark coordinates by subtracting the minimum x and y values
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))  # Store normalized x-coordinate
                    data_aux.append(y - min(y_))  # Store normalized y-coordinate

            # Ensure the extracted data matches the expected feature count before adding to dataset
            if len(data_aux) == EXPECTED_FEATURES:
                data.append(data_aux)  # Add processed data
                labels.append(dir_)  # Add corresponding label
            else:
                print(f"Skipped image {img_path} in {dir_}: incomplete data with {len(data_aux)} features.")

# Save the dataset as a pickle file for later use
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(f"Dataset saved. Total samples: {len(data)}")
