# -*- coding: utf-8 -*-
"""
FROM: https://yadi.sk/i/XueHw94udPoH9
BOW 2 KERNEL 
"""

from sklearn.svm import SVC
import nltk
import re
import string

ALF = [chr(i) for i in range(ord('a'), ord('z') + 1)] + [chr(i)
        for i in range(ord('A'), ord('Z') + 1)]


def dot(x, y):
    return sum(i[0] * i[1] for i in zip(x, y))

def make_collection(docs):
    text = ''    
    ''' слияние текстов '''
    for i in docs:
        text = text+i[:-2]+' '
    text = text[:-1]
    ''' теги чисел '''
    match = re.findall(r'[0-9]+[.,/]*[0-9]*', text)
    for i in sorted(match,key = len, reverse = True):                                       
        text = string.replace(text,i,' ')    
    ''' чистка символов '''   
    for i in ['(',')','-','/',',']:
        text = string.replace(text,i,' ') 
    ''' лишние \w '''
    match = re.findall(r' +',text)
    for i in match:
        text = string.replace(text,i,' ')
        
    return set(nltk.word_tokenize(text)) - STOP_WORDS

def make_dic(doc):
    dic = {}
    for word in doc.split():
        if word in dic:
            dic[word] += 1
        else:
            dic[word] = 1
    return dic
    
                
def make_vector(dic, collection):
    vector = []
    for word in collection:
        if word in dic:
            vector.append(dic[word])
        else:
            vector.append(0)
    return vector

def make_pars(docs):
    global collection
    return [make_vector(make_dic(doc), collection) for doc in docs]

def del_n(doc):
    return string.replace(doc[:-1], '/n', ' ')
    
def make_kernel(x, y):
    Kernel = []
    if x != y:
        for i in make_pars(x):
            t = []
            for j in make_pars(y):
                t.append(dot(i, j))
            Kernel.append(t)
    else:
        parse = make_pars(x)
        for i in parse:
            t = []
            for j in parse:
                t.append(dot(i, j))
            Kernel.append(t)
    return Kernel



''' reading data '''

STOP_WORDS = set([i.strip('\n') for i in open('stop_words.txt', 'r'
                 ).readlines()])

f = open('train_set_texts.txt', 'r')
X = []
y = []

for i in f:
    (label, text) = i.split('\t')
    X.append(text)
    y.append(label)

X_train = X
X_test = X[30:35]
y_train = y
y_test = y[30:35]

    

''' make collection of words '''

collection = []
for i in X:
    collection.extend(make_collection(i))

''' Creating model '''
model = SVC(kernel='precomputed')
   
''' loading kernel + fiting model '''

model.fit(make_kernel(X_train, X_train), y_train)

''' loading train set + predicting labels '''

y_predicted = model.predict(make_kernel(X_test, X_train))

n = 0
for i in range(len(y_predicted)):
    if y_test[i] == y_predicted[i]:
        n += 1

print n, len(y_predicted)