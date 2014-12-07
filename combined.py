# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 19:50:35 2014

@author: olga
"""

from sklearn.svm import SVC
from sklearn import cross_validation as cv
import re
import string
import nltk

l=0.5
n=14

alf='abcdefghijklmnopqrstuvwxyz'

def makeadic(a):
    pars=[]
    global alf, collection
    for i in a:
        dic={}
        text=nltk.word_tokenize(i)
        for i in text:
            if i[0] in alf:
                if i in dic:
                    dic[i]+=1
                else:
                    dic[i]=1
        vector=[]
        for i in collection1:
            if i in dic:
                vector.append(dic[i])
            else:
                vector.append(0)
        pars.append(vector)
    return pars

def makecollection(a):
    global alf
    pars=[]
    text=[]
    text1=nltk.word_tokenize(i)
    for j in text1:
        if j[0] in alf:
            text.append(j)
    text=set(text)
    text=text-stop_words
    pars.extend(list(text))
    return pars
    
    
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
        
    collection=[]
    for i in range(len(text)-n+1):
        collection.append(text[i:i+n])
    collection=list(set(collection))
    return collection


def make_list_vector(a):
    global collection
    pars=[]
    for doc in a:
        vector=[]
        for i in collection2:
            match=re.findall(i,doc)
            vector.append(len(match))
        pars.append(vector)
    return pars    
    
    
    
def dot(x,y):
    return sum(i[0]*i[1] for i in zip(x,y))    


### reading data ###

stop_words=[]
stop_words1=open('stop_words.txt', 'r').readlines()
for i in stop_words1:
    stop_words.append(i.strip('\n'))
stop_words=set(stop_words)    
    

f = open('train_set_texts.txt','r')
X = []
y = []

for i in f:
    label,text = i.split('\t')
    X.append(text)
    y.append(label)
    
'''
X=X[:40]
y=y[:40]    
'''    
    
 

    
### cross validation ###
X_train, X_test, y_train, y_test = cv.train_test_split(X, y, test_size=0.8, random_state=0)

### make collection of words ###
collection1=[]
for i in X_train:
    collection1.extend(makecollection(i))

collection2=make_collection(X_train, n) 

### Creating model ###
model = SVC(kernel='precomputed')
   
### loading kernel + fiting model ###
# loading
X_parsed1=makeadic(X_train)
X_parsed2=make_list_vector(X_train)
Kernel = []
for i in range(len(X_parsed1)):
    t = []
    for j in range(len(X_parsed1)):
        t.append(dot(X_parsed1[i],X_parsed1[j])*(1-l)+dot(X_parsed2[i],X_parsed2[j])*l)
    Kernel.append(t)
# fiting
model.fit(Kernel, y_train)

### loading train set + predicting labels ###
X_t_parsed1=makeadic(X_test)
X_t_parsed2=make_list_vector(X_test)
Kernel_t = []
for i in range(len(X_t_parsed1)):
    t = []
    for j in range(len(X_parsed1)):
        t.append(dot(X_t_parsed1[i],X_parsed1[j])*(1-l)+dot(X_t_parsed2[i],X_parsed2[j])*l)
    Kernel_t.append(t)
 
y_predicted=model.predict(Kernel_t)


n=0
for i in range(len(y_predicted)):
    if y_test[i]==y_predicted[i]:
        n+=1
    
print n, len(y_predicted)