from Face_recognition.MIX import main
from AI_Chatbot.Old_model import chatbot 
from voicetotext import speechRecognition as SR
from textTovoice.voice import text_to_speech
from textTovoice.voice import text_to_speech_speed
import time

person_name = main.main()
print (person_name)
text_to_speech_speed("I am UP....", 100 )
while True :
    recognizedText =  SR.speech()
    if person_name == "multiple":
        bot_response = "Hello Multiple people" 
    elif person_name != None:
        bot_response = chatbot.chatKnown(message=recognizedText, name= person_name)
    else:   
        bot_response = chatbot.chat(message=recognizedText)

    bot_reply = bot_response[0]
    reply_intent = bot_response[1] #Could be used for signal propogation only if reply_intent is "greeting"

    print('YOU : ',recognizedText)
    time.sleep(2)
    print('Bot : ', bot_reply)
    text_to_speech(bot_reply)