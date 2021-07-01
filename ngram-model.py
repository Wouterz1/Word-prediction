import re
import os
import time
from nltk import word_tokenize, pos_tag_sents, everygrams, pos_tag
from nltk.lm.api import Smoothing
from nltk.util import everygrams, pad_sequence
from nltk.lm import MLE, KneserNeyInterpolated
from nltk.lm.preprocessing import flatten
from nltk.lm.preprocessing import padded_everygram_pipeline
from collections import defaultdict, Counter
import dill 


ngram_size = 4

p = os.path.dirname(__file__)    
file = open(os.path.join(p, "unked-clean-dict-15k\\train_text.txt")).readlines()        #training data 
eval_file = open(os.path.join(p, "unked-clean-dict-15k\\eval_kss_en.txt")).readlines()  #evaluation data
eval_text = open(os.path.join(p, "eval_text.txt")).readlines()                          #grammatical correctness data
file = [line.lower() for line in file]
eval_file = [line.lower() for line in eval_file]
eval_text = [line.lower() for line in eval_text]

#tokenize with nltk
def tokenize(text):
    text = [re.sub('s>', '', line) for line in text]
    text = [re.sub(r'\d+', 'num', line) for line in text]
    text = [re.sub(r'\W+', ' ', line) for line in text]
    tokens = [word_tokenize(line) for line in text]
    return tokens

#pos tag the tokens
def pos_tagging(tokens):
    pos_tokens = pos_tag_sents(tokens)
    return pos_tokens

#create dict with occurence ngrams of the tokens
def make_ngrams(tokens):
    ngrams = defaultdict(lambda: defaultdict(lambda: 0))
    for sentence in tokens:
        for w4, w3, w2, w1 in everygrams(sentence, min_len=4, max_len=4, pad_right=True, pad_left=True): #make n-grams and pad them
            ngrams[(w4,w3,w2)][w1] += 1 
    return ngrams

#create vocab (not used)
def make_vocab(tokens):
    vocab = set(flatten(tokens))
    return vocab

#normalize ngram dict (not used)
def norm_dict(ngrams):
    for ngram in ngrams:
        total = float(sum(ngrams[ngram].values()))
        for word in ngrams[ngram]:
            ngrams[ngram][word] /= total

#train n-gram model
def train(tokens, name):
    data, vocab = padded_everygram_pipeline(4, tokens)  #preprocessing
    model = KneserNeyInterpolated(4)                    #KneserNey, cause best one according to majority
    model.fit(data, vocab)                              #train model
    with open(name +'.pkl', 'wb') as save:
        dill.dump(model, save)                          #save model

#load model
def load(name):         
    with open(name +'.pkl', 'rb') as load:
        model = dill.load(load)
    return model

#test model
def test(model):
    #x = 0
    #r = 0
    #while x < 100: #for repsonse time testing 
    while True:
        print("Type something (or type \'quit\' to stop):")
        #words = "this is a test to see"
        words = input().lower()
        words = re.sub(r'\d+', 'num', words)
        words = re.sub(r'\W+', ' ', words)
        if words == 'quit':
            break
        wordsSplit = words.split()
        #now = time.perf_counter_ns() 
        pred = model.generate(1, wordsSplit) #predict word
        #done = time.perf_counter_ns() 
        print('Guess: {}'.format(pred)) 
        #r += done-now  
        #x += 1
        #print(x)
    #print(r/x)

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

def test_with_pos(model, pos_ngrams):
    #x = 0
    #r = 0
    #while x < 100: #for repsonse time testing
    while True:
        print("Type something (or type \'quit\' to stop):")
        #words = "this is a test to see"
        words = input().lower()
        words = re.sub(r'\W+', ' ', words)
        if words == 'quit':
            break        
        #now = time.perf_counter_ns() 
        pred = gen(model, words.split(), pos_ngrams) #predict word
        #done = time.perf_counter_ns() 
        print('Guess: {}'.format(pred))
        #r += done-now  
        #x += 1
        #print(x)
    #print(r/x)

#calculate perplexity
def statistics(model, text):
    tokenized_text = list(tokenize(text))
    ngrams = [everygrams(t, min_len=1, max_len =4, pad_right=True, pad_left=True, left_pad_symbol="<s>", right_pad_symbol="</s>") for t in tokenized_text] #pre-process data so it fitl model
    total_pp = 0
    for ngram in ngrams:
        x = model.perplexity(ngram) #calculated perplexity
        total_pp += x
        print("PP: {}".format(x))
    print('Total PP: {}'.format(total_pp/len(ngrams))) #average perplexity over the whole data
    input().lower()
    
#calculate accuracy
def accuracy(model, text):    
    tokenized_text = tokenize(text)
    correct = 0
    incorrect = 0    
    for n in range(500):
        for sent in tokenized_text:
            for i in range(len(sent)-2):
                wordsSplit = sent[:i+1] #get all words up till now 
                pred = model.generate(1, wordsSplit) #generate next word from last three
                if pred == sent[i+2]:
                    correct += 1
                else:
                    incorrect += 1
        acc = correct/(correct+incorrect)
        print(acc)
    return acc

#calculate accuracy pos
def accuracy_pos(model, text, pos_ngrams):  
    text = [re.sub(r'\W+', ' ', line) for line in text]
    tokenized_text = [word_tokenize(line) for line in text]
    correct = 0
    incorrect = 0
    for n in range(500):
        for sent in tokenized_text:
            for i in range(len(sent)-4):
                wordsSplit = sent[:i+3] #get all words up till now
                pred = gen(model, wordsSplit, pos_ngrams) #generate next word
                if pred == sent[i+4]:
                    correct += 1
                else:
                    incorrect += 1
        acc = correct/(correct+incorrect)
        print(acc)
    return acc

#generate word for accuracy and grammatical correctness
def gen(model, inp, pos_ngrams):
    words = re.sub(r'\d+', 'num', ' '.join(inp))
    predictions = set()
    x = 0
    while len(predictions) < 5 and x < 20:
        predictions.add(model.generate(1, words.split())) #generate next word
        x+=1
    for word in predictions:
        sent = ' '.join(inp)+' '+word
        pos = pos_tag(word_tokenize(sent)) #pos tag sentence
        if pos[-1][1] in get_best_pos(pos_ngrams, (pos[-4][1],pos[-3][1],pos[-2][1])): #check if pos tag is common
            return word
    pred = model.generate(1, words.split())
    if pred == '</s>' or pred == 'num':
            x = 0
            while (pred == '</s>' or pred == 'num') and x < 10:
                pred = model.generate(1, words.split())
                x+=1
    return pred

#results for grammatical correctness
def gram_test(model, eval_text):
    text = tokenize(eval_text)
    for sent in text:
        pred = model.generate(1, sent)
        if pred == '</s>' or pred == 'num':
            x = 0
            while (pred == '</s>' or pred == 'num') and x < 10:
                pred = model.generate(1, sent)
                x+=1
        print('{} ({})'.format(' '.join(sent), pred))
    return

#results for grammatical correctness pos
def gram_test_pos(model, eval_text, pos):
    text = [re.sub(r'\W+', ' ', line) for line in eval_text]
    text = [word_tokenize(line) for line in text]
    for sent in text:
        pred = gen(model, sent, pos)
        print('{} ({})'.format(' '.join(sent), pred))
    return


x = tokenize(file)                                          #pre-process training data
y = pos_tagging(x)                                          #pos tag data
pos_ngrams = make_pos_ngrams(y)                             #make pos ngrams
name = "kney_model"                                         #model name 
#train(x, name)                                             #train model using training data
model = load(name)                                          #load model
#per = statistics(model, eval_file)                         #get perplexity
#acc = accuracy(model, eval_file)                           #get accuracy
#acc_poss = accuracy_pos(model, eval_file, pos_ngrams)      #get accuracy pos
#test(model)                                                #type stuff and model returns predictions per input
#test_with_pos(model, pos_ngrams)                           #type stuff and pos model returns predictions per input
gram_test(model, eval_text)                                 #grammatical correctness results
print('-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-')  #Idk, some output divider again 
gram_test_pos(model, eval_text, pos_ngrams)                 #grammatical correctness results
input().lower()                                             #enter to exit program




