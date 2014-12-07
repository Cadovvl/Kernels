# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 20:59:54 2014

@author: olga

BOW 2 KERNEL test=0.8 0.945
"""
from sklearn.svm import SVC
from sklearn import cross_validation as cv
import nltk

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
        for i in collection:
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
    
    

    
### cross validation ###
X_train, X_test, y_train, y_test = cv.train_test_split(X, y, test_size=0.8, random_state=0)

### make collection of words ###
collection=[]
for i in X:
    collection.extend(makecollection(i))

### Creating model ###
model = SVC(kernel='precomputed')
   
### loading kernel + fiting model ###
# loading
X_parsed=makeadic(X_train)
Kernel = []
for i in X_parsed:
    t = []
    for j in X_parsed:
        t.append(dot(i,j))
    Kernel.append(t)
# fiting
model.fit(Kernel, y_train)

### loading train set + predicting labels ###
X_t_parsed=makeadic(X_test)
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