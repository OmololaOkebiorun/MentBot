import nltk
from nltk.stem import WordNetLemmatizer
lema = WordNetLemmatizer()
import pickle
import numpy as np

from keras.models import load_model
model = load_model('mentbot_model.h5')
import json
import random
intents = json.loads(open('mentalhealth.json', encoding='utf-8').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))


#defining a function that will clean user input
def clean_user_input(user_input):
    user_input_words = nltk.word_tokenize(user_input)
    user_input_words = [lema.lemmatize(word.lower()) for word in user_input_words]
    return user_input_words


#defining a function that will return bag of words - an numpy array that is a representative of the input
def bag_of_words(sentence, words, show_details=True):
    sentence_words = clean_user_input(sentence)    # tokenize the input with clean_user_input function defined above
    bag = [0]*len(words)  #  creates a list of 0s of lenght of list classes
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1 # assign 1 if current word is in the vocabulary position
                if show_details:
                    print ("found in bag: %s" % w)
    return np.array(bag)


#defining a function that will predict the class of tag the user input belongs
def sentence_class(sentence, model):
    prediction = bag_of_words(sentence, words, show_details=False)
    result = model.predict(np.array([prediction]), verbose = 0)[0] #returns an array of the probability of prediction
    limit = 0.25 #setting a threshold of 0.25
    results = [[i,r] for i,r in enumerate(result) if r > limit]# filter out predictions below the limit
    results.sort(key=lambda x: x[1], reverse=True)# sort by probability
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list #returns the class with highest probability


#defining a function that will get response of the tag with the highest probability from the data
def Response(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents: #looping through the data to find the tag with the highest probability and returning a random response 
        if(i['tag']== tag): 
            result = random.choice(i['responses'])
            break
        else:
            result = "You must ask the right questions" #if tag is not found
    return result


def chatbot_response(msg):
    ints = sentence_class(msg, model)
    res = Response(ints, intents)
    return res


