# -*- coding: utf-8 -*-
"""
FROM: https://yadi.sk/i/XueHw94udPoH9
SSK+BOW 
"""

import re
from sklearn.svm import SVC
import numpy as np
import string


def make_collection_bow(docs):
    
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



def make_vectors_bow(docs):
    global collectionBow
    pars = []
    for doc in docs:
        vector = []
        for word in collectionBow:
            expr = re.compile(word)
            vector.append(len(expr.findall(doc)))
        pars.append(vector)
    return pars


def make_collection_ssk(docs, n):
    collection = []
    for doc in docs:
        for i in ['\n', '(', ')', '/', '\\', '.', '^', '$', '*', '+', '?', '{', '}', '[', ']', '|']:
            doc = string.replace(doc, i, ' ')
        stringList = []
        for i in range(len(doc) - n + 1):
            text = ''
            for j in range(n):
                text += doc[i+j]
            stringList.append(text)
        
        collection.extend(stringList)
    
    return list(set(collection))
    
    
def make_vectors_ssk(docs):
    global collectionSsk
    pars = []
    for doc in docs:
        vector = []
        for stri in collectionSsk:
            expr = re.compile(stri)
            vector.append(len(expr.findall(doc)))
        pars.append(vector)
    return pars

STOP_WORDS = set([i.strip('\n') for i in open('stop_words.txt', 'r'
                 ).readlines()])

n= 17
l=0.5

### reading data ###
X_train = [open('DATA/DP/TrainSet/train{0}.txt'.format(i)).read() for i in xrange(190,208) ]
y_train = [i.strip() for i in open('DATA/DP/train_set_lables.txt').readlines()]

X_test = [open('DATA/DP/TestSet/test{0}.txt'.format(i)).read() for i in xrange(790, 812) ]
y_test = [i.strip() for i in open('DATA/DP/test_set_lables.txt').readlines()]

# making bag of words #
collectionBow = make_collection_bow(X_train)
collectionSsk = make_collection_ssk(X_train, n)

# making vectors #
vectorsTrainBow = make_vectors_bow(X_train)
vectorsTestBow = make_vectors_bow(X_test)

vectorsTrainSsk = make_vectors_ssk(X_train)
vectorsTestSsk = make_vectors_ssk(X_test)

# Creating model #
model = SVC(kernel='precomputed')

# loading kernel + fiting model #


# for BOW #
kernelTrainBow = []
for vector in vectorsTrainBow:
    t = []
    for vector1 in vectorsTrainBow:
        t.append(np.dot(vector, vector1))
    kernelTrainBow.append(t)
    
kernelTrainBow = np.array(kernelTrainBow)


# for SSK #
kernelTrainSsk = []
for vector in vectorsTrainSsk:
    t = []
    for vector1 in vectorsTrainSsk:
        t.append(np.dot(vector, vector1))
    kernelTrainSsk.append(t)
    
kernelTrainSsk = np.array(kernelTrainSsk)
    
kernelTrainCombined = (1-l) * kernelTrainBow + l * kernelTrainSsk


model.fit(kernelTrainCombined, y_train)


# loading train set + predicting labels #

# for BOW #
kernelTestBow = []
for vector in vectorsTestBow:
    t = []
    for vector1 in vectorsTrainBow:
        t.append(np.dot(vector, vector1))
    kernelTestBow.append(t)
    
kernelTestBow = np.array(kernelTestBow)    


# for SSK #
kernelTestSsk = []
for vector in vectorsTestSsk:
    t = []
    for vector1 in vectorsTrainSsk:
        t.append(np.dot(vector, vector1))
    kernelTestSsk.append(t)
    
kernelTestSsk = np.array(kernelTestSsk)
    
kernelTestCombined = (1-l) * kernelTestBow + l * kernelTestSsk

y_predicted = model.predict(kernelTestCombined)

# score #

n = 0
for i in range(len(y_predicted)):
    if y_test[i] == y_predicted[i]:
        n += 1
        
print n, len(y_predicted)