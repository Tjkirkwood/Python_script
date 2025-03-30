import cv2
import mediapipe as mp
import pyautogui
import numpy as np
from filterpy.kalman import KalmanFilter
import torch

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Get screen size
screen_width, screen_height = pyautogui.size()

# Open webcam with wider FoV (if supported)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 60)  # Higher FPS for smoother tracking

# Kalman Filter for smoother motion
kf = KalmanFilter(dim_x=4, dim_z=2)
kf.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
kf.P *= 1000
kf.R *= 5
kf.Q *= 0.1
kf.x = np.array([0, 0, 0, 0])

# Check if CUDA is available for GPU acceleration
use_gpu = torch.cuda.is_available()
print("GPU Acceleration:", "Enabled" if use_gpu else "Not Available")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip and convert color
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process frame (use GPU if available)
    with torch.no_grad():
        results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get index finger tip coordinates
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x, y = int(index_finger_tip.x * screen_width), int(index_finger_tip.y * screen_height)
            
            # Kalman filter prediction and update
            kf.predict()
            kf.update([x, y])
            x_smooth, y_smooth = kf.x[:2]
            
            # Move cursor smoothly
            pyautogui.moveTo(int(x_smooth), int(y_smooth), duration=0.01)
            
            # Detect pinch gesture (thumb and index close together)
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            distance = abs(index_finger_tip.x - thumb_tip.x) + abs(index_finger_tip.y - thumb_tip.y)
            
            if distance < 0.05:
                pyautogui.click()
                cv2.putText(frame, "Click!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Show frame with tracking
    cv2.imshow("Hand Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
