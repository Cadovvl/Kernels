# -*- coding: utf-8 -*-
"""
FROM: http://wseob.ru/seo/mashinnoe-obuchenie-classify
	  https://yadi.sk/i/lUzV6SypdPoGz
SSK 
"""

from sklearn.svm import SVC
import re
import string

n = 28
def make_collection(a, n):
    text = ''    
    
    ### слияние текстов ###
    for i in a:
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
        
    collection = []
    for i in range(len(text) - n+1):
        collection.append(text[i:i+n])
    collection = list(set(collection))
    return collection

def make_vector(a):
    global collection
    pars = []
    for doc in a:
        vector = []
        for i in collection:
            match = re.findall(i, doc)
            vector.append(len(match))
        pars.append(vector)
    return pars


def dot(x, y):
    return sum(i[0] * i[1] for i in zip(x, y))


def make_kernel(x, y):
    Kernel = []
    if x != y:
        for i in make_vector(x):
            t = []
            for j in make_vector(y):
                t.append(dot(i, j))
            Kernel.append(t)
    else:
        parse = make_vector(x)
        for i in parse:
            t = []
            for j in parse:
                t.append(dot(i, j))
            Kernel.append(t)
    return Kernel


### reading data ###

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

collection = make_collection(X_train, n)

### Creating model ###

model = SVC(kernel='precomputed')

### loading kernel + fiting model ###

# fiting

model.fit(make_kernel(X_train, X_train), y_train)

### loading train set + predicting labels ###

y_predicted = model.predict(make_kernel(X_test, X_train))

n = 0
for i in range(len(y_predicted)):
    if y_test[i] == y_predicted[i]:
        n += 1

print n, len(y_predicted)
