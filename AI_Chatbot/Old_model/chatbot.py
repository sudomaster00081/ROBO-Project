import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()


class ChatBot:
    def __init__(self):
        self.intents = json.loads(open("AI_Chatbot/Old_model/intents.json").read())
        self.words = pickle.load(open('AI_Chatbot/Old_model/words.pkl', 'rb'))
        self.classes = pickle.load(open('AI_Chatbot/Old_model/classes.pkl', 'rb'))
        self.model = load_model("AI_Chatbot/Old_model/chatbotmodel.h5")

    def clean_up_sentence(self, sentence):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
        return sentence_words

    def bag_of_words(self, sentence):
        sentence_words = self.clean_up_sentence(sentence)
        bag = [0] * len(self.words)
        for w in sentence_words:
            for i, word in enumerate(self.words):
                if word == w:
                    bag[i] = 1
        return np.array(bag)

    def predict_class(self, sentence):
        bow = self.bag_of_words(sentence)
        res = self.model.predict(np.array([bow]))[0]
        ERROR_THRESHOLD = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({'intent': self.classes[r[0]], 'probability': str(r[1])})
        return return_list

    def get_response(self, ints):
        result = "I don't understand!"
        try:
            tag = ints[0]['intent']
            list_of_intents = self.intents['intents']
            for i in list_of_intents:
                if i['tag'] == tag:
                    result = random.choice(i['responses'])
                    break
        except IndexError:
            pass
        return result


    def request(self, message):
        ints = self.predict_class(message)
        response = self.get_response(ints)
        return response, ints


def chat(message : str):
    chatbot = ChatBot()

    #message = input("")
    message = message.lower()
    res = chatbot.request(message)
    botReply = res[0]
    return botReply

def chatKnown(message : str, name):
    chatbot = ChatBot()

    #message = input("")
    message = message.lower()
    res = chatbot.request(message)
    botReply = res[0] 
    replyIntent = res[1][0]

    if(replyIntent['intent']=='greeting'):
        botReply += " " + name 

    return botReply


#chatbot.py driver code
if __name__ == '__main__':
    while True:
        chatbot = ChatBot()
        message = input("You: ")
        #message = input("")
        message = message.lower()
        res = chatbot.request(message)

        botReply = res[0]
        replyIntent = res[1][0]

        name = "Person"
        if(replyIntent['intent']=='greeting'):
            botReply += " " + name
        
        print("bot: ", botReply)