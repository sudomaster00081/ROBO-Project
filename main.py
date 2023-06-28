from Face_recognition.MIX import main
from AI_Chatbot.Old_model import chatbot 
from voicetotext import speechRecognition as SR
from textTovoice.voice import text_to_speech
from textTovoice.voice import text_to_speech_speed
import time

person_name = main.main()
text_to_speech_speed("I am UP....", 100 )
while True :
    recognizedText = SR.speech()
    if person_name != None:
        bot_response = chatbot.chatKnown(message=recognizedText, name= person_name)
    else:   
        bot_response = chatbot.chat(message=recognizedText)

    print('YOU : ',recognizedText)
    time.sleep(2)
    print('Bot : ', bot_response)
    text_to_speech(bot_response)