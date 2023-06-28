import pyttsx3

#TEXT TO SPEECH
def text_to_speech(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 140)  # Speed of speech
    engine.setProperty('volume', 1.0)  # Volume level (0.0 to 1.0)
    engine.say(text)
    engine.runAndWait()


def text_to_speech_speed(text, speed):
    engine = pyttsx3.init()
    engine.setProperty('rate', speed)  # Speed of speech
    engine.setProperty('volume', 1.0)  # Volume level (0.0 to 1.0)
    engine.say(text)
    engine.runAndWait()
