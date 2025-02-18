import cv2
import mediapipe as mp
import numpy as np
import pickle

# Load the trained model
with open('model.p', 'rb') as f:
    model_data = pickle.load(f)
    model = model_data['model']

# Load label classes
label_classes = np.load('label_classes.npy')

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Helper function to extract hand landmarks from an image
def extract_hand_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)
    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.append(lm.x)
            landmarks.append(lm.y)
        return landmarks
    return None

# Capture from webcam for real-time inference
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Extract landmarks
    landmarks = extract_hand_landmarks(frame)
    
    if landmarks:
        # Ensure the data input size matches what the model expects (42 features)
        data_aux = np.array(landmarks).reshape(1, -1)  # Reshape to match input size

        # Make prediction
        prediction = model.predict(data_aux)
        predicted_class = chr(prediction[0] + ord('A'))  # Convert back to letter A-Z
        
        # Display the predicted letter on the frame
        cv2.putText(frame, predicted_class, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Show the frame
    cv2.imshow('Hand Sign Prediction', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
