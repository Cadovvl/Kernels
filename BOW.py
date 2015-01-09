# -*- coding: utf-8 -*-
"""
FROM: https://yadi.sk/i/XueHw94udPoH9
BOW 2 KERNEL 
"""

import re
from sklearn.svm import SVC
import numpy as np


def make_collection (docs):
    
    exprTire = re.compile('[a-z]+-[a-z]+')
    exprWords = re.compile('[a-z]+')

    collection = []
    
    for doc in docs:
        
        # finding words #
        wordsList = exprWords.findall(doc)
        tireList = exprTire.findall(doc)
        
        tireParsed = []
        for i in range(len(tireList)):
            tireParsed.extend(tireList[i].split('-'))
 
        # del tire #
        for i in tireParsed:
            if i in wordsList:
                del wordsList[wordsList.index(i)]
                
        wordsList += tireList
        partCollection = list(set(wordsList) - STOP_WORDS)
        
        collection.extend(partCollection)
    
    return list(set(collection))



def make_vectors(docs):
    global collection
    pars = []
    for doc in docs:
        vector = []
        for word in collection:
            expr = re.compile(word)
            vector.append(len(expr.findall(doc)))
        pars.append(vector)
    return pars



STOP_WORDS = set([i.strip('\n') for i in open('stop_words.txt', 'r'
                 ).readlines()])

### reading data ###
X_train = [open('DATA/DATA1/80-20/TrainSet/train{0}.txt'.format(i)).read() for i in xrange(1600) ]
y_train = [i.strip() for i in open('DATA/DATA1/80-20/train_set_lables.txt').readlines()]

X_test = [open('DATA/DATA1/80-20/TestSet/test{0}.txt'.format(i)).read() for i in xrange(400) ]
y_test = [i.strip() for i in open('DATA/DATA1/80-20/test_set_lables.txt').readlines()]

# making bag of words #
collection = make_collection(X_train)

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