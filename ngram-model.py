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
file = open(os.path.join(p, "unked-clean-dict-15k\\train_text.txt")).readlines()
eval_file = open(os.path.join(p, "unked-clean-dict-15k\\eval_kss_en.txt")).readlines()
file = [line.lower() for line in file]
eval_file = [line.lower() for line in eval_file]

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
        for w4, w3, w2, w1 in everygrams(sentence, min_len=4, max_len=4, pad_right=True, pad_left=True):
            ngrams[(w4,w3,w2)][w1] += 1 
    return ngrams

#create vocab (not used atm)
def make_vocab(tokens):
    vocab = set(flatten(tokens))
    return vocab

#normalize ngram dict
def norm_dict(ngrams):
    for ngram in ngrams:
        total = float(sum(ngrams[ngram].values()))
        for word in ngrams[ngram]:
            ngrams[ngram][word] /= total

def train(tokens, name):
    data, vocab = padded_everygram_pipeline(4, tokens)
    model = KneserNeyInterpolated(4)
    model.fit(data, vocab)
    with open(name +'.pkl', 'wb') as save:
        dill.dump(model, save)
        
def load(name):
    with open(name +'.pkl', 'rb') as load:
        model = dill.load(load)
    return model

def test(model):
    while True:
        print("Type something (or type \'quit\' to stop):")
        words = input().lower()
        words = re.sub(r'\d+', 'num', words)
        words = re.sub(r'\W+', ' ', words)
        if words == 'quit':
            break
        wordsSplit = words.split()
        now = time.perf_counter_ns() 
        pred = model.generate(1, wordsSplit)
        done = time.perf_counter_ns() 
        print('Guess: {}'.format(pred))
        print('Response time: {}'.format(done - now))

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
    #while x < 100: 
    while True:
        print("Type something (or type \'quit\' to stop):")
        #words = "this is a test to see"
        words = input().lower()
        words = re.sub(r'\W+', ' ', words)
        if words == 'quit':
            break        
        now = time.perf_counter_ns() 
        pred = gen(model, words.split(), pos_ngrams)
        done = time.perf_counter_ns() 
        print('Guess: {}'.format(pred))
        print('Response time: {}'.format(done - now))
        #r += done-now  
        #x += 1
        #print(x)
    #print(r/x)

def statistics(model, text):
    tokenized_text = list(tokenize(text))
    ngrams = [everygrams(t, min_len=1, max_len =4, pad_right=True, pad_left=True, left_pad_symbol="<s>", right_pad_symbol="</s>") for t in tokenized_text]
    total_pp = 0
    for ngram in ngrams:
        x = model.perplexity(ngram)
        total_pp += x
        print("PP: {}".format(x))
    print('Total PP: {}'.format(total_pp/len(ngrams)))
    input().lower()
    
def accuracy(model, text):    
    tokenized_text = tokenize(text)
    correct = 0
    incorrect = 0
    
    for n in range(500):
        for sent in tokenized_text:
            for i in range(len(sent)-2):
                wordsSplit = sent[:i+1]
                pred = model.generate(1, wordsSplit)
                if pred == sent[i+2]:
                    correct += 1
                else:
                    incorrect += 1
        acc = correct/(correct+incorrect)
        print(acc)
    return acc

def accuracy_pos(model, text, pos_ngrams):  
    text = [re.sub(r'\W+', ' ', line) for line in text]
    tokenized_text = [word_tokenize(line) for line in text]
    correct = 0
    incorrect = 0
    for n in range(500):
        for sent in tokenized_text:
            for i in range(len(sent)-4):
                wordsSplit = sent[:i+3]
                pred = gen(model, wordsSplit, pos_ngrams)
                if pred == sent[i+4]:
                    correct += 1
                else:
                    incorrect += 1
        acc = correct/(correct+incorrect)
        print(acc)
    return acc

def gen(model, inp, pos_ngrams):
    words = re.sub(r'\d+', 'num', ' '.join(inp))
    predictions = set()
    x = 0
    while len(predictions) < 5 and x < 20:
        predictions.add(model.generate(1, words.split()))
        x+=1
    for word in predictions:
        sent = ' '.join(inp)+' '+word
        pos = pos_tag(word_tokenize(sent))
        if pos[-1][1] in get_best_pos(pos_ngrams, (pos[-4][1],pos[-3][1],pos[-2][1])):
            return word
    return model.generate(1, words.split())


x = tokenize(file)
y = pos_tagging(x)
pos_ngrams = make_pos_ngrams(y)
name = "kney_model"
#train(x, name)
model = load(name)
#per = statistics(model, eval_file)
#acc = accuracy(model, eval_file)
#acc_poss = accuracy_pos(model, eval_file, pos_ngrams)
#test(model)
test_with_pos(model, pos_ngrams)
input().lower()




