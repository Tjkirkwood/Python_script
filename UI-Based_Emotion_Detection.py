import cv2
from deepface import DeepFace
import numpy as np
import threading
import tkinter as tk
from PIL import Image, ImageTk

# Initialize Tkinter UI
root = tk.Tk()
root.title("AI Emotion Detector")

# Set window size
root.geometry("800x600")
root.configure(bg="black")

# Label to display the emotion
emotion_label = tk.Label(root, text="Emotion: Detecting...", font=("Arial", 24), fg="white", bg="black")
emotion_label.pack(pady=20)

# Video display area
video_label = tk.Label(root)
video_label.pack()

# OpenCV video capture
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

def detect_emotion(frame):
    """ Detect emotion from the given frame """
    try:
        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # DeepFace analysis
        analysis = DeepFace.analyze(rgb_frame, actions=["emotion"], enforce_detection=False)
        
        if analysis and "dominant_emotion" in analysis[0]:
            return analysis[0]["dominant_emotion"]
    except Exception as e:
        print(f"Error detecting emotion: {e}")
    
    return "Unknown"

def update_video():
    """ Capture video frame, detect emotion, and update UI """
    ret, frame = cap.read()
    
    if ret:
        # Flip for natural interaction
        frame = cv2.flip(frame, 1)
        
        # Run emotion detection in a separate thread
        emotion = detect_emotion(frame)
        
        # Update label with detected emotion
        emotion_label.config(text=f"Emotion: {emotion}")

        # Convert OpenCV image to Tkinter format
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(img)

        # Display video frame in Tkinter UI
        video_label.config(image=img)
        video_label.image = img

    root.after(30, update_video)

# Start video update loop
update_video()

# Run Tkinter event loop
root.mainloop()

# Release webcam when closing
cap.release()
cv2.destroyAllWindows()
