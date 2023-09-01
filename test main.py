
from Face_recognition.MIX import main

from AI_Chatbot.Old_model import chatbot 
from AI_Chatbot.Old_model.intentLib import intentAction

from voicetotext import speechRecognition as SR

from textTovoice.voice import text_to_speech
from textTovoice.voice import text_to_speech_speed

import time


# import threading
# from flask import Flask, render_template
# from Face_recognition.MIX import main as face_recognition
# from AI_Chatbot.Old_model import chatbot
# from AI_Chatbot.Old_model.intentLib import intentAction
# from voicetotext import speechRecognition as SR
# from textTovoice.voice import text_to_speech

# app = Flask(__name__)
# person_name = None
# bot_response = None

# @app.route('/')
# def index():
#     return render_template('index.html', person_name=person_name, bot_response=bot_response)

# def recognize_face():
#     global person_name
#     person_name = face_recognition.main()
#     if person_name == "multiple":
#         person_name = "Multiple people"
#     app.run(debug=False, use_reloader=False)


# if __name__ == '__main__':
#     # Start Flask app in a separate thread
#     web_thread = threading.Thread(target=recognize_face)
#     web_thread.start()

#     # Start chatbot in the main thread


from flask import Flask, render_template
import threading

app = Flask(__name__)

# Your chatbot logic can be implemented here
chatbot_data = []
person_name = 'unknown'

@app.route('/')
def index():
    return render_template('index.html', chatbot_data=chatbot_data, user = person_name)

def flask_thread():
    app.run()

#Flask LOGG Manager
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

if __name__ == '__main__':
    flask_thread = threading.Thread(target=flask_thread)
    flask_thread.start()
    # You can now add logic to populate chatbot_data from the main thread.
    # For example: chatbot_data.append("Message from main thread")

    person_name = main.main()
    #print (person_name)

    if person_name == "multiple":
            bot_resp = "Hello Multiple people"
            text_to_speech(bot_resp)
            
    text_to_speech_speed("I am UP....", 100 )

    while True :
        recognizedText =  SR.speech()
        chatbot_data.append(recognizedText)
        if person_name != None and person_name != "multiple":
            bot_response = chatbot.chatKnown(message=recognizedText, name= person_name)
            
        else:   
            bot_response = chatbot.chat(message=recognizedText)
            

        #Taking bot reply and related intent from tuple
        bot_reply = bot_response[0]
        chatbot_data.append(bot_reply)
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