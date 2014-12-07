# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 22:30:21 2014

@author: olga

NGRAM KERNEL N=3 TEST=0.8 0.8075
             N=2 TEST=0.8 0.9225
"""

from sklearn.svm import SVC
from sklearn import cross_validation as cv
import re
import string
import nltk

n=2

def make_collection(a, n):
    text=''    
    
    ### слияние текстов ###
    for i in a:
        text=text+i[:-2]+' '
    text=text[:-1]
    ### теги чисел ###
    match=re.findall(r'[0-9]+[.,/]*[0-9]*', text)
    for i in sorted(match,key=len, reverse=True):                                       
        text=string.replace(text,i,'num')    
    ### чистка символов ###   
    for i in ['(',')','-','/',',']:
        text=string.replace(text,i,' ') 
    ### лишние \w ###
    match=re.findall(r' +',text)
    for i in match:
        text=string.replace(text,i,' ')
    
    text=nltk.word_tokenize(text)
    
    collection=[]
    for i in range(len(text)-n+1):
        collection.append(text[i:i+n])
        
    pars=[]
    for i in collection:
        if i not in pars:
            pars.append(i)
        
    return pars


def make_list_vector(a):
    global collection
    pars=[]
    for doc in a:
        vector=[]
        for i in collection:
            text=''
            for j in i:
                text+=j+' '
            match=re.findall(text[:-1],doc)
            vector.append(len(match))
        pars.append(vector)
    return pars
            


def dot(x,y):
    return sum(i[0]*i[1] for i in zip(x,y))    


### reading data ###


f = open('train_set_texts.txt','r')
X = []
y = []

for i in f:
    label,text = i.split('\t')
    X.append(text)
    y.append(label)
    
'''
X=X[:10]
y=y[:10]    
'''
    
### cross validation ###
X_train, X_test, y_train, y_test = cv.train_test_split(X, y, test_size=0.8, random_state=0)



### Creating model ###
model = SVC(kernel='precomputed')

### Making collection ###
collection=make_collection(X_train, n)   
### loading kernel + fiting model ###
# loading

X_parsed=make_list_vector(X_train)
Kernel = []
for i in X_parsed:
    t = []
    for j in X_parsed:
        t.append(dot(i,j))
    Kernel.append(t)
# fiting
model.fit(Kernel, y_train)

### loading train set + predicting labels ###
X_t_parsed=make_list_vector(X_test)
Kernel_t = []
for i in X_t_parsed:
    t = []
    for j in X_parsed:
        t.append(dot(i,j))
    Kernel_t.append(t)
 
y_predicted=model.predict(Kernel_t)


n=0
for i in range(len(y_predicted)):
    if y_test[i]==y_predicted[i]:
        n+=1
    
print n, len(y_predicted)