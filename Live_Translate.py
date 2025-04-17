import easyocr
import mss
import numpy as np
import cv2
from langdetect import detect
from googletrans import Translator
import time
import threading

# Initialize EasyOCR (for English and Japanese in this case)
reader = easyocr.Reader(['en', 'ja'], gpu=False)  # You can change languages if needed

# Initialize Google Translate
translator = Translator()

# Global variable to store selected region and monitor
selected_region = None
primary_monitor = None

# Function to capture the screen region from the primary monitor
def capture_screen():
    with mss.mss() as sct:
        while True:
            if selected_region is not None and primary_monitor is not None:
                screenshot = sct.grab(primary_monitor)  # Use only the primary monitor
                screenshot_image = np.array(screenshot)
                yield screenshot_image
            time.sleep(0.1)

# Function to extract text using EasyOCR
def extract_text(image):
    result = reader.readtext(image)
    return [(text, box) for _, text, box in result]

# Function to detect language and translate the detected text (phrase)
def translate_text(text):
    try:
        detected_language = detect(text)
        if detected_language != 'en':  # Translate if the text isn't English
            translated = translator.translate(text, src='auto', dest='en')
            return translated.text
        return text
    except Exception as e:
        print(f"Error with translation: {e}")
        return text

# Function to overlay translated text on the live view
def display_overlay(image, text, position=(10, 10)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, text, position, font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    return image

# Function to select a screen region (using OpenCV's ROI tool)
def select_region():
    print("Please select a region on the screen...")

    # Capture a screenshot of the primary monitor to display for region selection
    with mss.mss() as sct:
        screenshot = sct.grab(primary_monitor)  # Capture the entire screen of the primary monitor
        screenshot_image = np.array(screenshot)

    # Let the user select the region using OpenCV's selectROI
    region = cv2.selectROI("Select Region", screenshot_image)
    cv2.destroyWindow("Select Region")

    if region[2] == 0 or region[3] == 0:
        print("Invalid region selected. Please select a valid region with non-zero width and height.")
        return select_region()
    return region

# Function to initialize the primary monitor (hard-coded for 1080p monitor)
def initialize_primary_monitor():
    global primary_monitor
    with mss.mss() as sct:
        # Check for monitor with 1080p resolution (1920x1080)
        for monitor in sct.monitors[1:]:
            if monitor['width'] == 1920 and monitor['height'] == 1080:
                primary_monitor = monitor
                print(f"Primary monitor selected: {primary_monitor}")
                return primary_monitor
    print("No 1080p monitor found. Selecting the first monitor by default.")
    primary_monitor = sct.monitors[1]  # Fallback to first monitor
    return primary_monitor

# Function to process screen and show translations live
def process_screen():
    global selected_region
    initialize_primary_monitor()  # Initialize primary monitor
    selected_region = select_region()  # Let the user select the capture region

    for screenshot_image in capture_screen():
        # Extract text from the screen region
        texts = extract_text(screenshot_image)

        for text, _ in texts:
            if text.strip():  # Ensure the text isn't empty
                print(f"Detected text: {text}")
                translated_text = translate_text(text)
                print(f"Translated text: {translated_text}")

                # Display the translated text as overlay
                screenshot_image = display_overlay(screenshot_image, translated_text)

        # Display the live view with the overlay
        cv2.imshow("Live Translation", screenshot_image)
        cv2.waitKey(1)  # Update frame every loop

# Start the process in a separate thread for real-time updates
def main():
    # Start the processing function in a separate thread
    thread = threading.Thread(target=process_screen)
    thread.daemon = True
    thread.start()

    try:
        while True:
            time.sleep(1)  # Keep the program running
    except KeyboardInterrupt:
        print("Exiting program...")

if __name__ == "__main__":
    main()
