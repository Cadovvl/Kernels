# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 17:09:47 2014

@author: olga

4 SSK test=0.8 n=8 0.9375
n=14
"""

from sklearn.svm import SVC
from sklearn import cross_validation as cv
import re
import string

n=14

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
        for i in collection:
            match=re.findall(i,doc)
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
    

X=X[:50]
y=y[:50]    

    
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