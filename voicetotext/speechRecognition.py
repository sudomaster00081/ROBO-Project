import speech_recognition as sr

def speech():
    # create a recognizer instance
    r = sr.Recognizer()

    # use the default microphone as the audio source
    with sr.Microphone() as source:
        print("Speak now...")
        # adjust for ambient noise
        r.adjust_for_ambient_noise(source)
        # listen for audio input
        audio = r.listen(source)

    try:
        # recognize speech using Google Speech Recognition
        text = r.recognize_google(audio)
        return text
        #print("You said: ", text)
        
        # check if the "stop" command was spoken
        # if text.lower() == "stop":
        #     print("Stopping the script...")
        #     break
        
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))
    return "none"

if __name__ == '__main__':
        
    # create a recognizer instance
    r = sr.Recognizer()


    # use the default microphone as the audio source
    with sr.Microphone() as source:
        print("Speak now...")
        # adjust for ambient noise
        r.adjust_for_ambient_noise(source)
        # listen for audio input
        audio = r.listen(source)

    try:
        # recognize speech using Google Speech Recognition
        text = r.recognize_google(audio)
        print("You said: ", text)
        
        # # check if the "stop" command was spoken
        # if text.lower() == "stop":
        #     print("Stopping the script...")
        #     break
        
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))