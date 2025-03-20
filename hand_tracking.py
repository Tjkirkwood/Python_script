# Author: Tyler K.

import cv2
import mediapipe as mp
import pyautogui
import math

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.9, min_tracking_confidence=0.9)

# Get screen size
screen_width, screen_height = pyautogui.size()

# Open webcam
cap = cv2.VideoCapture(0)

# Set webcam resolution (optional for wider view)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Set width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Set height

# Smoothing variables (Exponential Moving Average)
prev_x, prev_y = -1, -1
alpha = 0.7  # Smoothing factor: higher = smoother
threshold = 5  # Minimum distance change for cursor to move

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip and convert color
    frame = cv2.flip(frame, 1)
    
    # Preprocessing: Convert to grayscale and apply histogram equalization
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    equalized_frame = cv2.equalizeHist(gray_frame)  # Histogram equalization to improve contrast
    rgb_frame = cv2.cvtColor(equalized_frame, cv2.COLOR_GRAY2RGB)  # Convert back to RGB for MediaPipe
    
    # Process frame
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the full frame
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get index finger tip coordinates
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x, y = int(index_finger_tip.x * screen_width), int(index_finger_tip.y * screen_height)
            
            # Apply Exponential Moving Average (EMA) smoothing
            if prev_x == -1 and prev_y == -1:
                smoothed_x, smoothed_y = x, y  # Initialize on the first frame
            else:
                smoothed_x = alpha * x + (1 - alpha) * prev_x
                smoothed_y = alpha * y + (1 - alpha) * prev_y

            # Ensure coordinates are within screen bounds
            smoothed_x = max(0, min(screen_width - 1, smoothed_x))
            smoothed_y = max(0, min(screen_height - 1, smoothed_y))

            # Calculate movement direction
            delta_x = smoothed_x - prev_x
            delta_y = smoothed_y - prev_y

            # Move cursor based on direction (X, Y movement)
            if abs(delta_x) > threshold or abs(delta_y) > threshold:
                pyautogui.moveRel(delta_x, delta_y)  # Move relative to current position
            
            # Update previous coordinates
            prev_x, prev_y = smoothed_x, smoothed_y

            # Detect pinch gesture (thumb and index close together)
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            # Calculate Euclidean distance between thumb and index tip
            distance = math.sqrt((index_finger_tip.x - thumb_tip.x) ** 2 + (index_finger_tip.y - thumb_tip.y) ** 2)
            
            if distance < 0.03:  # Tighter threshold for pinch detection
                pyautogui.click()
                cv2.putText(frame, "Click!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show frame with tracking
    cv2.imshow("Hand Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
