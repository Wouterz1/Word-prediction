import re
import os
from tensorflow.keras.backend import categorical_crossentropy
import time
import numpy
import tensorflow as tf
from nltk import word_tokenize, pos_tag_sents, pos_tag
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Embedding, Activation
from collections import defaultdict
import pickle


ngram_size = 4

p = os.path.dirname(__file__)    
file = open(os.path.join(p, "train_text.txt")).read().lower()                           #training data 
file_val = open(os.path.join(p, "valid_text.txt")).read().lower()                       #validation data 
file_sents = open(os.path.join(p, "train_text.txt")).readlines()                        #training data ngrams
eval_text = open(os.path.join(p, "eval_text.txt")).readlines()                          #evaluation data for gram. correctness
file_eval = open(os.path.join(p, "unked-clean-dict-15k\\eval_kss_en.txt")).readlines()  #evaluation data

#tokenize with nltk
def tokenize(text):    
    text = re.sub('s>', '', text)
    text = re.sub(r'\d+', 'num', text)
    text = re.sub(r'\W+', ' ', text)
    tokens = word_tokenize(text)
    return tokens

#create dict with 'token : number' pair for all unique tokens
def make_numbers(tokens):
    token_dict= {}
    count = 1
    for i in range(len(tokens)):
        if tokens[i] not in token_dict:
            token_dict[tokens[i]] = count
            count += 1
    return token_dict

#create ngrams of the tokens
def make_ngrams(tokens):
    ngrams = []
    for i in range((len(tokens)-ngram_size)+1):
        prev = tokens[i:i+ngram_size]
        ngrams.append(prev)
    return ngrams

#turn words into individual numbers, each a unique one we can refer to later. We do this so we can use them in vectors
def make_numbers_ngrams(ngrams):
    tok = Tokenizer(oov_token='oov')                 
    tok.fit_on_texts(ngrams)     #make words individual integers
    ngrams_int = tok.texts_to_sequences(ngrams) #turn the words in the ngrams into the integers
    return ngrams_int, tok

#create matrix with all "words"
def make_matrices(ngram_int, tok):
    matrix = numpy.empty([len(ngram_int), ngram_size]) #create empty numpy array
    for i in range(len(ngram_int)):
        matrix[i] = ngram_int[i]    #fill the numpy array
    input = matrix[:,:-1]           #input is first 3 words of ngram
    targets = matrix[:,-1]          #target is last word of the ngram
    targets = to_categorical(targets, num_classes=len(tok.word_counts)+2)   #one-hot vectorization (e.g. ngram (3, 4, 1) -> ([00010], [00001], [01000]) )
    return input, targets

#make validation data
def make_val(text, tok):
    text = [line.lower() for line in text]
    text = [re.sub('s>', '', line) for line in text]
    text = [re.sub(r'\d+', 'num', line) for line in text]
    text = [re.sub(r'\W+', ' ', line) for line in text]
    tokens_sents = [word_tokenize(line) for line in text]

    ngrams = []
    for sent in tokens_sents:
        for i in range((len(sent)-ngram_size)+1):
            prev = sent[i:i+ngram_size]
            ngrams.append(prev)

    n_ngrams = tok.texts_to_sequences(ngrams)

    valx, valy = make_matrices(n_ngrams, tok)
    return valx, valy

def training(input, targets, valx, valy, tok, name):
    my_model = Sequential()
    my_model.add(Embedding(len(tok.word_counts)+2, input.shape[1], input_length=input.shape[1]))
    my_model.add(LSTM(256, return_sequences=True))  #lstm
    my_model.add(LSTM(256))  #lstm
    my_model.add(Dense(256, activation='relu')) #add dense NN layer to speed up
    my_model.add(Dense(len(tok.word_counts)+2)) #add dense layer for output
    my_model.add(Activation('softmax')) #normalize output

    my_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy') #use adam because recommended for large data
    #hist = my_model.fit(input, targets, validation_data=(valx, valy), epochs=1000) should work too, but no time to use
    hist = my_model.fit(input, targets, epochs=1000)
    pickle.dump(hist, open("history.p", "wb"))
    my_model.save(name)
    return

def test(model, tok):
    #x = 0
    #r = 0
    #while x < 101: # test for average response time
    while True:
        print("Type something (or type \'quit\' to stop):")
        #words = "this is a test to see"
        words = input().lower()
        words = re.sub(r'\d+', 'num', words)
        words = re.sub(r'\W+', ' ', words)
        if words == 'quit':
            break
        #now = time.perf_counter_ns() 
        words = tok.texts_to_sequences([words])[0] #parse words to intergers again so model understands
        padded_words = pad_sequences([words], maxlen=ngram_size-1) #pad the sequences in case they are too short for he model to handle
        predictions = model.predict(padded_words)[0].argsort()[-3:][::-1] #take top 3 predictions
        #done = time.perf_counter_ns() 
        pred_words = []
        for i in predictions:
            pred_words.append(tok.index_word[i]) #make predictions readable for us
        #if x > 0:
            #r += done-now
        #x += 1
        print(pred_words)
        #print(done-now)
    #print(r/x)
    return

#pos tag the tokens
def pos_tag(text):
    text = [re.sub('s>', '', line) for line in text]
    text = [re.sub(r'\W+', ' ', line) for line in text]
    tokens = [word_tokenize(line) for line in text]
    pos_tokens = pos_tag_sents(tokens)
    return pos_tokens

#make pos ngrams
def make_pos_ngrams(pos_tokens):
    pos_ngrams = defaultdict(lambda: defaultdict(lambda: 0))
    for sent in pos_tokens:
        for i in range((len(sent)-ngram_size)+1):
            array = []
            for word, pos in sent[i:i+ngram_size]:
                array.append(pos)
            pos_ngrams[(array[0],array[1],array[2])][array[3]] += 1 #count the occurence of each of the ngrams
    return pos_ngrams

#get the pos-tags that occur most frequent after the initial 3 tags
def get_best_pos(pos_ngrams, ngram):
    current = pos_ngrams[ngram]
    best_pos = [key for (key, value) in current.items() if value > max(current.values())/2]
    return best_pos

def test_with_pos(model, tok, pos_ngrams):
    #x = 0
    #r = 0
    #while x < 101:  # test for average response time
    while True:
        print("Type something (or type \'quit\' to stop):")
        words = input().lower()
        #words = "this is a test to see"
        words = re.sub(r'\W+', ' ', words)
        if words == 'quit':
            break
        #now = time.perf_counter_ns() 
        words_num = tok.texts_to_sequences([re.sub(r'\d+', 'num', words)])[0] #parse words to intergers again so model understands
        padded_words = pad_sequences([words_num], maxlen=ngram_size-1) #pad the sequences in case they are too short for the model to handle
        predictions = model.predict(padded_words)[0].argsort()[-5:][::-1] #take top 5 predictions
        pred_words = []
        for i in predictions:
            pred_words.append(tok.index_word[i])
        best_pred = []
        for word in pred_words:
            pos = pos_tag((words+' '+word).split()) #pos tag our sequence + prediction
            if pos[-1][0][1] in get_best_pos(pos_ngrams, (pos[-4][0][1],pos[-3][0][1],pos[-2][0][1])): #see if pos is one that occurs frequently
                #done = time.perf_counter_ns()
                #if x > 0:
                #    r += done-now 
                best_pred.append(word)
                #print(done-now)
                if len(best_pred) >= 3:
                    print(best_pred)
        if len(best_pred) < 3:
            for word in pred_words:
                if word not in best_pred:
                    best_pred.append(word)                    
                    if len(best_pred) == 3:
                        print(best_pred)
        #x += 1
    #print(r/(x-1))
    return

def get_evaluation(model, tok, text):
    # tokenize for sentence
    text = [line.lower() for line in text]
    text = [re.sub('s>', '', line) for line in text]
    text = [re.sub(r'\d+', 'num', line) for line in text]
    text = [re.sub(r'\W+', ' ', line) for line in text]
    tokens_sents = [word_tokenize(line) for line in text]

    #make ngrams
    ngrams = []
    for sent in tokens_sents:
        for i in range((len(sent)-ngram_size)+1):
            prev = sent[i:i+ngram_size]
            ngrams.append(prev)

    n_ngrams = tok.texts_to_sequences(ngrams) # words -> integers

    eval_input, eval_targets = make_matrices(n_ngrams, tok) #integers -> input and target (one-hot)
    eval_values = model.evaluate(x=eval_input, y=eval_targets)
    return eval_values[0] #returns cross-entropy

#calc accuracy of model
def get_acc(model, file_eval, tok):
    correct = 0
    incorrect = 0
    text = [tokenize(sent) for sent in file_eval]
    for sent in text:
        words = tok.texts_to_sequences([sent])[0] #parse words to intergers again so model understands
        for i in range(len(words)-4):
            word = words[:i+3]
            padded_words = pad_sequences([word], maxlen=ngram_size-1) #pad the sequences in case they are too short for the model to handle
            prediction = tok.index_word[model.predict(padded_words)[0].argsort()[:-2][::-1]]
            if prediction[0] == 'num':
                prediction = prediction[1]
            if prediction == sent[i+4]:
                correct += 1
            else:
                incorrect += 1
        print(correct+incorrect)
    acc = correct/(correct+incorrect)
    print(acc)
    return

#get pred for accuracy and gram correctness with pos 
def get_pred(pos_ngrams, preds, tok, word):
    pred_words = []
    for i in preds:
        pred_words.append(tok.index_word[i])
    for w in pred_words:
        pos = pos_tag((' '.join(word)+' '+w).split())
        if pos[-1][0][1] in get_best_pos(pos_ngrams, (pos[-4][0][1],pos[-3][0][1],pos[-2][0][1])): #see if pos is one that occurs frequently
            return w
    if pred_words[0] == 'num':
        return pred_words[1]
    return pred_words[0]

#calc accuracy of model with pos
def get_acc_pos(model, file_eval, tok, pos_ngrams):
    correct = 0
    incorrect = 0
    text = [word_tokenize(re.sub(r'\W+', ' ', sent)) for sent in file_eval]
    for sent in text:
        words = tok.texts_to_sequences([(re.sub(r'\d+', 'num', ' '.join(sent))).split()])[0] #parse words to intergers again so model understands
        for i in range(len(words)-4):
            word = words[:i+3]
            padded_words = pad_sequences([word], maxlen=ngram_size-1)  #pad the sequences in case they are too short for the model to handle
            predictions = model.predict(padded_words)[0].argsort()[-5:][::-1]
            prediction = get_pred(pos_ngrams, predictions, tok, sent[:i+3])
            if prediction == sent[i+4]:
                correct += 1
            else:
                incorrect += 1
        print(correct+incorrect)
    acc = correct/(correct+incorrect)
    print(acc)
    return

#test gram correctness
def gram_test(model, file_eval, tok):
    text = [tokenize(sent) for sent in file_eval]
    for sent in text:
        words = tok.texts_to_sequences([sent])[0] #parse words to intergers again so model understands
        padded_words = pad_sequences([words], maxlen=ngram_size-1)  #pad the sequences in case they are too short for the model to handle
        prediction = tok.index_word[model.predict(padded_words)[0].argsort()[-1]]
        print('{} ({})'.format(' '.join(sent), prediction))
    return

#test gram correctness pos tagger
def gram_test_pos(model, file_eval, tok, pos_ngrams):
    text = [re.sub(r'\W+', ' ', line) for line in file_eval]
    text = [word_tokenize(line) for line in text]
    for sent in text:
        words = tok.texts_to_sequences([(re.sub(r'\d+', 'num', ' '.join(sent))).split()])[0] #parse words to intergers again so model understands
        padded_words = pad_sequences([words], maxlen=ngram_size-1)  #pad the sequences in case they are too short for the model to handle
        predictions = model.predict(padded_words)[0].argsort()[-5:][::-1]
        prediction = get_pred(pos_ngrams, predictions, tok, sent)
        print('{} ({})'.format(' '.join(sent), prediction))
    return

tokens = tokenize(file)                                     #training data
pos = pos_tag(file_sents)                                   #pos tag training data
pos_ngrams = make_pos_ngrams(pos)                           #ngrams of pos
#token_dict = make_numbers(tokens)                          #not used
ngrams = make_ngrams(tokens)                                #ngrams for training data
num_grams, tokenizer = make_numbers_ngrams(ngrams)          #transform training data into integers with tokenizer
#inp, targ = make_matrices(num_grams, tokenizer)            #transform training data into input and targets (one-hot vectors) for model
name = 'my_word_predictor_4'                                #name of model
#valx, valy = make_val(file_val, tokenizer)                #make validation data for model
#training(inp, targ, valx, valy, tokenizer, name)           #train model

model = load_model(name)                                    #load trained model
#get_acc(model, file_eval, tokenizer)                       #get accuracy scores for model
#get_acc_pos(model, file_eval, tokenizer, pos_ngrams)       #get accuracy scores for model with pos
gram_test(model, eval_text, tokenizer)                      #get results for gram. correctness
print("-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-") #Idk, divider for the lines
gram_test_pos(model, eval_text, tokenizer, pos_ngrams)      #get results for gram. correctness with pos
#pp = get_evaluation(model, tokenizer, file_eval)           #get perplexity of model
#print('PP: {}'.format(tf.exp(pp), acc))                    #print perplexity
#test_with_pos(model, tokenizer, pos_ngrams)                #type stuff and model returns predictions per input
#test(model, tokenizer)                                     #type stuff and pos model returns predictions per input
input()                                                     #enter to stop program

