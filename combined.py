# -*- coding: utf-8 -*-
"""
FROM: https://yadi.sk/i/XueHw94udPoH9
SSK+BOW 
"""

from sklearn.svm import SVC
import re
import string
import nltk

l = 0.5
n = 14

ALF = [chr(i) for i in range(ord('a'), ord('z') + 1)] + [chr(i)
        for i in range(ord('A'), ord('Z') + 1)]


def dot(x, y):
    return sum(i[0] * i[1] for i in zip(x, y))

def make_collection1(docs):
    text = ''    
    ### слияние текстов ###
    for i in docs:
        text = text+i[:-2]+' '
    text = text[:-1]
    ### теги чисел ###
    match = re.findall(r'[0-9]+[.,/]*[0-9]*', text)
    for i in sorted(match,key = len, reverse = True):                                       
        text = string.replace(text,i,' ')    
    ### чистка символов ###   
    for i in ['(',')','-','/',',']:
        text = string.replace(text,i,' ') 
    ### лишние \w ###
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


def make_vector1(dic, collection):
    vector = []
    for word in collection:
        if word in dic:
            vector.append(dic[word])
        else:
            vector.append(0)
    return vector


def make_pars(docs):
    global collection1
    return [make_vector1(make_dic(doc), collection1) for doc in docs]


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




def clean_text(docs):
    text = ''    
    ### слияние текстов ###
    for i in docs:
        text = text+i[:-2]+' '
    text = text[:-1]
    ### теги чисел ###
    match = re.findall(r'[0-9]+[.,/]*[0-9]*', text)
    for i in sorted(match,key = len, reverse = True):                                       
        text = string.replace(text,i,'num')    
    ### чистка символов ###   
    for i in ['(',')','-','/',',']:
        text = string.replace(text,i,' ') 
    ### лишние \w ###
    match = re.findall(r' +',text)
    for i in match:
        text = string.replace(text,i,' ')
    return text.split()

def make_collection2(docs, n):
    collection = []
    for i in range(len(clean_text(docs)) - n + 1):
        collection.append(clean_text(docs)[i:i + n])

    pars = []
    for i in collection:
        if i not in pars:
            pars.append(i)

    return pars


def make_vector2(a):
    global collection2
    pars = []
    for doc in a:
        vector = []
        for i in collection2:
            text = ''
            for j in i:
                text += j + ' '
            match = re.findall(text[:-1], doc)
            vector.append(len(match))
        pars.append(vector)
    return pars


### reading data ###

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



### make collection of words ###

collection1 = []
for i in X_train:
    collection1.extend(make_collection1(i))

collection2 = make_collection2(X_train, n)

### Creating model ###

model = SVC(kernel='precomputed')

### loading kernel + fiting model ###
# loading

X_parsed1 = make_pars(X_train)
X_parsed2 = make_vector2(X_train)
Kernel = []
for i in range(len(X_parsed1)):
    t = []
    for j in range(len(X_parsed1)):
        t.append(dot(X_parsed1[i], X_parsed1[j]) * (1 - l)
                 + dot(X_parsed2[i], X_parsed2[j]) * l)
    Kernel.append(t)

# fiting

model.fit(Kernel, y_train)

### loading train set + predicting labels ###

X_t_parsed1 = make_pars(X_test)
X_t_parsed2 = make_vector2(X_test)
Kernel_t = []
for i in range(len(X_t_parsed1)):
    t = []
    for j in range(len(X_parsed1)):
        t.append(dot(X_t_parsed1[i], X_parsed1[j]) * (1 - l)
                 + dot(X_t_parsed2[i], X_parsed2[j]) * l)
    Kernel_t.append(t)

y_predicted = model.predict(Kernel_t)

n = 0
for i in range(len(y_predicted)):
    if y_test[i] == y_predicted[i]:
        n += 1

print n, len(y_predicted)
