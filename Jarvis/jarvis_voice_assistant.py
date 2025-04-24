import speech_recognition as sr
import pyttsx3
import openai

# 🔑 Your OpenAI API key
openai.api_key = "YOUR_OPENAI_API_KEY"

# 🎙️ Initialize speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 170)
engine.setProperty('voice', 'com.apple.speech.synthesis.voice.alex' if 'macOS' in engine.getProperty('voices')[0].id else engine.getProperty('voices')[0].id)

def speak(text):
    print("Jarvis:", text)
    engine.say(text)
    engine.runAndWait()

def listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎤 Listening...")
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)
    try:
        command = r.recognize_google(audio)
        print("You:", command)
        return command
    except sr.UnknownValueError:
        speak("Sorry, I didn't catch that.")
        return ""
    except sr.RequestError:
        speak("Speech recognition service is down.")
        return ""

def chat_with_gpt(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # or "gpt-3.5-turbo"
            messages=[{"role": "user", "content": prompt}]
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        print("OpenAI Error:", e)
        return "Sorry, I had an issue with the AI service."

def main():
    speak("Jarvis online. Ready for your command.")
    while True:
        query = listen()
        if query.lower() in ["exit", "quit", "shutdown"]:
            speak("Shutting down. Goodbye.")
            break
        elif query:
            response = chat_with_gpt(query)
            speak(response)

if __name__ == "__main__":
    main()
