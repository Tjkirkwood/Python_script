import easyocr
import mss
import time
import threading
import numpy as np
import cv2
from langdetect import detect
from googletrans import Translator
import pyautogui


# Initialize the OCR reader (EasyOCR)
reader = easyocr.Reader(['ja', 'en'], gpu=False)

# Initialize Google Translator for text translation
translator = Translator()

# Global variable to store selected region coordinates
selected_region = None
is_dragging = False
start_point = None
end_point = None

# Function to capture mouse events for interactive region selection
def select_region(event, x, y, flags, param):
    global is_dragging, start_point, end_point, selected_region

    if event == cv2.EVENT_LBUTTONDOWN:
        # Start selecting the region
        is_dragging = True
        start_point = (x, y)
        end_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        # Update the region as the mouse moves
        if is_dragging:
            end_point = (x, y)
            temp_img = np.copy(screenshot_image)  # Create a copy of the image for temporary display
            cv2.rectangle(temp_img, start_point, end_point, (0, 255, 0), 2)  # Draw rectangle while dragging
            cv2.imshow("Select Region", temp_img)

    elif event == cv2.EVENT_LBUTTONUP:
        # Finish selecting the region
        is_dragging = False
        end_point = (x, y)
        selected_region = (start_point[0], start_point[1], end_point[0] - start_point[0], end_point[1] - start_point[1])
        print(f"Region selected: {selected_region}")

# Function to extract text from the screenshot
def extract_text(image):
    results = reader.readtext(image)
    texts = [(text, box) for _, text, box in results]  # Texts and their bounding boxes
    return texts

# Function to detect the language of the text using langdetect
def detect_language(text):
    try:
        language = detect(text)
        return language
    except Exception as e:
        print(f"Error in language detection: {e}")
        return None

# Function to translate text using Google Translator
def translate_text(text):
    try:
        translated = translator.translate(text, src='auto', dest='en')  # Translate to English
        return translated.text
    except Exception as e:
        print(f"Error in translation: {e}")
        return text  # Return original text if there's an error

# Function to capture and process screen region (runs in a separate thread)
def capture_and_process():
    global selected_region

    with mss.mss() as sct:
        while selected_region is None:
            # Keep displaying the region selection window until the user selects the region
            cv2.imshow("Select Region", screenshot_image)
            cv2.waitKey(1)

        # Capture the selected region
        monitor = {"top": selected_region[1], "left": selected_region[0], "width": selected_region[2], "height": selected_region[3]}
        while True:
            screenshot = sct.grab(monitor)
            screenshot_image = np.array(screenshot)  # Convert to a NumPy array for EasyOCR

            # Extract texts from the screenshot
            texts = extract_text(screenshot_image)

            # Process each text found
            for text, _ in texts:
                print(f"Detected text: {text}")
                detected_language = detect_language(text)
                if detected_language != 'en':  # If it's not already in English
                    translated_text = translate_text(text)
                    print(f"Translated text: {translated_text}")
                else:
                    print(f"No translation needed: {text}")

            time.sleep(1)  # Delay to prevent excessive CPU usage

# Function to initialize the region selection
def main():
    global screenshot_image
    screenshot = pyautogui.screenshot()
    screenshot_image = np.array(screenshot)  # Convert screenshot to numpy array for OpenCV
    screenshot_image = cv2.cvtColor(screenshot_image, cv2.COLOR_RGB2BGR)

    # Display the initial screenshot and allow region selection
    cv2.imshow("Select Region", screenshot_image)
    cv2.setMouseCallback("Select Region", select_region)

    # Wait for user to finish selecting the region
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Start OCR and translation process in a separate thread
    translation_thread = threading.Thread(target=capture_and_process)
    translation_thread.daemon = True  # Ensure the thread stops when the main program ends
    translation_thread.start()

    try:
        while True:
            time.sleep(1)  # Keep main thread alive
    except KeyboardInterrupt:
        print("Exiting program...")

# Run the program
if __name__ == "__main__":
    main()
