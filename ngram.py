# -*- coding: utf-8 -*-
"""
FROM: https://yadi.sk/i/6Y0SlAhMdQpFZ
NGRAM KERNEL
"""

import re
from sklearn.svm import SVC
import numpy as np


def make_collection(docs, n):
    exprWords = re.compile('[a-z]+')

    collection = []
    
    for doc in docs:        
        # finding words #
        wordsList = exprWords.findall(doc)
        
        ngramList = []
        for i in range(len(wordsList) - n + 1):
            text = ''
            for j in range(n):
                text += wordsList[i+j]+' '
            ngramList.append(text[:-1])
        
        collection.extend(ngramList)
    
    return list(set(collection))
        
def make_vectors(docs):
    global collection
    pars = []    
    for doc in docs:
                
        # making cleaned text #
        exprWords = re.compile('[a-z]+')
        wordsList = exprWords.findall(doc)
        text = ''
        for i in wordsList:
            text += i + ' '
            
            
        # making vectors #
        vector = []
        for ngram in collection:
            expr = re.compile(ngram)
            vector.append(len(expr.findall(text)))
            
        pars.append(vector)
    return pars
    
    
n = 3

### reading data ###
X_train = [open('DATA/DP/TrainSet/train{0}.txt'.format(i)).read() for i in xrange(190,208) ]
y_train = [i.strip() for i in open('DATA/DP/train_set_lables.txt').readlines()]

X_test = [open('DATA/DP/TestSet/test{0}.txt'.format(i)).read() for i in xrange(790, 812) ]
y_test = [i.strip() for i in open('DATA/DP/test_set_lables.txt').readlines()]


collection = make_collection(X_train, n)

# making vectors #
vectorsTrain = make_vectors(X_train)
vectorsTest = make_vectors(X_test)

# Creating model #
model = SVC(kernel='precomputed')
   
# loading kernel + fiting model #

kernelTrain = []
for vector in vectorsTrain:
    t = []
    for vector1 in vectorsTrain:
        t.append(np.dot(vector, vector1))
    kernelTrain.append(t)
    

model.fit(kernelTrain, y_train)

# loading train set + predicting labels #

kernelTest = []
for vector in vectorsTest:
    t = []
    for vector1 in vectorsTrain:
        t.append(np.dot(vector, vector1))
    kernelTest.append(t)
    
    
y_predicted = model.predict(kernelTest)

# score #

n = 0
for i in range(len(y_predicted)):
    if y_test[i] == y_predicted[i]:
        n += 1
        
print n, len(y_predicted)