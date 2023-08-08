from Face_recognition.MIX import main

from AI_Chatbot.Old_model import chatbot 
from AI_Chatbot.Old_model.intentLib import intentAction

from voicetotext import speechRecognition as SR

from textTovoice.voice import text_to_speech
from textTovoice.voice import text_to_speech_speed

import time


person_name = main.main()
#print (person_name)

if person_name == "multiple":
        bot_resp = "Hello Multiple people"
        text_to_speech(bot_resp)
        
text_to_speech_speed("I am UP....", 100 )

while True :
    recognizedText =  SR.speech()
    if person_name != None and person_name != "multiple":
        bot_response = chatbot.chatKnown(message=recognizedText, name= person_name)
    else:   
        bot_response = chatbot.chat(message=recognizedText)

    #Taking bot reply and related intent from tuple
    bot_reply = bot_response[0]
    reply_intent = bot_response[1] # intent trigger
    
    #function to perform action assigned to each intent
    try:
        intentfunction = getattr(intentAction, reply_intent)
        intentfunction()
    except AttributeError:
        pass
        
    
    print('YOU : ',recognizedText)
    time.sleep(2)
    print('Bot : ', bot_reply)
    text_to_speech(bot_reply)