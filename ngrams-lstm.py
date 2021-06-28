import re
import os
from tensorflow.keras.backend import categorical_crossentropy
import time
import numpy
from nltk import word_tokenize, pos_tag_sents, pos_tag
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Embedding, Activation
from collections import defaultdict
import pickle


ngram_size = 4
batch_size = 64

p = os.path.dirname(__file__)    
file = open(os.path.join(p, "unked-clean-dict-15k\\train_text.txt")).read().lower()
#file2 = open(os.path.join(p, "unked-clean-dict-15k\\test_text.txt")).read().lower()
file_sents = open(os.path.join(p, "unked-clean-dict-15k\\train_text.txt")).readlines()
#file_eval = open(os.path.join(p, "unked-clean-dict-15k\\eval_kss_en.txt")).readlines()
#tokenize with nltk
def tokenize(text):    
    text = re.sub('s>', '', file)
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
    tok = Tokenizer(oov_token='oov', num_words=15000)                 
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


def training(input, targets, tok, name):
    my_model = Sequential()
    my_model.add(Embedding(len(tok.word_counts)+2, input.shape[1], input_length=input.shape[1]))
    my_model.add(LSTM(256, return_sequences=True))  #lstm
    my_model.add(LSTM(256))  #lstm
    my_model.add(Dense(256, activation='relu')) #add dense NN layer to speed up
    my_model.add(Dense(len(tok.word_counts)+2)) #add dense layer for output
    my_model.add(Activation('softmax')) #normalize output

    my_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy') #use adam because recommended for large data
    my_model.fit(input, targets, epochs=1000)
    my_model.save(name)

def test(model, tok):
    #while True:
    x = 0
    r = 0
    while x < 100: 
        print("Type something (or type \'quit\' to stop):")
        words = "this is a test to see"
        #words = input().lower()
        words = re.sub(r'\d+', 'num', words)
        words = re.sub(r'\W+', ' ', words)
        if words == 'quit':
            break
        now = time.perf_counter_ns() 
        words = tok.texts_to_sequences([words])[0]
        padded_words = pad_sequences([words], maxlen=ngram_size-1)  
        predictions = model.predict(padded_words)[0].argsort()[-3:][::-1]
        done = time.perf_counter_ns() 
        pred_words = []
        for i in predictions:
            pred_words.append(tok.index_word[i])
        x += 1
        r += done-now
        print(pred_words)
        print(done-now)
    print(r/x)

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

#get the pos-tags that occur most frequent after the initial 3
def get_best_pos(pos_ngrams, ngram):
    current = pos_ngrams[ngram]
    best_pos = [key for (key, value) in current.items() if value > max(current.values())/2]
    return best_pos

def test_with_pos(model, tok, pos_ngrams):
    #x = 0
    #r = 0
    #while x < 101: 
    while True:
        print("Type something (or type \'quit\' to stop):")
        words = input().lower()
        #words = "this is a test to see"
        words = re.sub(r'\W+', ' ', words)
        if words == 'quit':
            break
        now = time.perf_counter_ns() 
        words_num = tok.texts_to_sequences([re.sub(r'\d+', 'num', words)])[0]
        padded_words = pad_sequences([words_num], maxlen=ngram_size-1)  
        predictions = model.predict(padded_words)[0].argsort()[-7:][::-1]
        pred_words = []
        for i in predictions:
            pred_words.append(tok.index_word[i])
        best_pred = []
        for word in pred_words:
            pos = pos_tag((words+' '+word).split())
            if pos[-1][0][1] in get_best_pos(pos_ngrams, (pos[-4][0][1],pos[-3][0][1],pos[-2][0][1])):
                done = time.perf_counter_ns()
                #if x > 0:
                #    r += done-now 
                best_pred.append(word)
                print(done-now)
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

def get_evaluation(model, tok, text):
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
    print(n_ngrams)

    eval_input, eval_targets = make_matrices(n_ngrams, tok)
    eval_values = model.evaluate(x=eval_input, y=eval_targets)
    return eval_values[0], eval_values[1]

tokens = tokenize(file)
pos = pos_tag(file_sents)
pos_ngrams = make_pos_ngrams(pos)
#token_dict = make_numbers(tokens)
ngrams = make_ngrams(tokens)
num_grams, tokenizer = make_numbers_ngrams(ngrams)
#inp, targ = make_matrices(num_grams, tokenizer)
name = 'my_word_predictor_3'
#training(inp, targ, tokenizer, name)

model = load_model(name)
#pp, acc = get_evaluation(model, tokenizer, file_eval)
test_with_pos(model, tokenizer, pos_ngrams)
test(model, tokenizer)
input()

