import pocketsphinx

def voice_to_text():
    config = pocketsphinx.Decoder.default_config()
    config.set_string('-hmm', 'path_to_acoustic_model')
    config.set_string('-lm', 'path_to_language_model')
    config.set_string('-dict', 'path_to_dictionary')

    decoder = pocketsphinx.Decoder(config)

    with open('audio_file.wav', 'rb') as audio_file:
        decoder.start_utt()
        while True:
            buf = audio_file.read(1024)
            if buf:
                decoder.process_raw(buf, False, False)
            else:
                break
        decoder.end_utt()

    hypothesis = decoder.hyp()
    if hypothesis is not None:
        return hypothesis.hypstr
    else:
        return ""

# Example usage
result = voice_to_text()
print("Transcribed Text:", result)
