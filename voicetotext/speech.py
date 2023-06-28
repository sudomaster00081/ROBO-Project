import speech_recognition as sr

def voice_to_text():
    r = sr.Recognizer()
    r.energy_threshold = 4000  # Adjust the value according to your audio input
    indian_english_model_path = "voicetotext/cmusphinx-en-in-5.2"

    # Set paths to Indian English acoustic and language models
    config = {
        'hmm': f'{indian_english_model_path}/en_in.cd_cont_5000',
        'lm': f'{indian_english_model_path}/en-us.lm.bin',
        'dict': f'{indian_english_model_path}/en_in.dic'
    }
    r.recognize_sphinx.set_property('pocketsphinx', config)

    with sr.Microphone() as source:
        print("Speak now...")
        audio = r.listen(source)  # Listen for audio input

    try:
        text = r.recognize_sphinx(audio, language='en-IN')
        return text
    except sr.UnknownValueError:
        print("Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Speech Recognition service; {0}".format(e))

# Example usage
result = voice_to_text()
print("Transcribed Text:", result)
