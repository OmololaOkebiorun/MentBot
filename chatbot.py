#importing the libraries needed
import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random

#creating an instance of WordNetLemmatizer
lemma = WordNetLemmatizer()

#loading the json file 
words=[]
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('mentalhealth.json', encoding='utf-8').read()
intents = json.loads(data_file)

#creating words and classes objects
for intent in intents['intents']:
    for pattern in intent['patterns']:

        w = nltk.word_tokenize(pattern)
        words.extend(w)

        documents.append((w, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemma.lemmatize(w.lower()) for w in words if w not in ignore_words] #changes the list in words to lower case
words = sorted(list(set(words))) # sorts the list (words)
classes = sorted(list(set(classes))) #sort the list (classes)

#serialize and save words and classes to a file
pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))


# initializing training data
training = []
output_empty = [0] * len(classes)#creates a list of 0s of lenght of list classes
for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemma.lemmatize(word.lower()) for word in pattern_words] # lemmatizes and changes to lower case

    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0) #appends words to list bag


    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

#introducing randomness into the training set and returing an array
random.shuffle(training)
training = np.array(training)

# creating train and test lists
train_x = list(training[:,0])
train_y = list(training[:,1])


#modelling 
model = Sequential() #creates a sequential model
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu')) #first layer (a dense layer with 128 units and relu activation function)
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu')) #second layer - dense layer with 64 units with relu activation function
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax')) #third layer - a dense layer with softmax activation function

# compiling model with stochastic gradient descent optimizer
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#fitting and saving the model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=0)
model.save('mentbot_model.h5', hist)

