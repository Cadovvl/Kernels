# -*- coding: utf-8 -*-
"""
FROM: http://wseob.ru/seo/mashinnoe-obuchenie-classify
	  https://yadi.sk/i/lUzV6SypdPoGz
SSK 
"""

import re
from sklearn.svm import SVC
import numpy as np
import string


def make_collection(docs, n):
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
    
    
def make_vectors(docs):
    global collection
    pars = []
    for doc in docs:
        vector = []
        for expr in regCollection:
            vector.append(len(expr.findall(doc)))
        pars.append(vector)
    return pars

n = 12

### reading data ###
X_train = [open('DATA/DP/TrainSet/train{0}.txt'.format(i)).read() for i in xrange(190,208) ]
y_train = [i.strip() for i in open('DATA/DP/train_set_lables.txt').readlines()]

X_test = [open('DATA/DP/TestSet/test{0}.txt'.format(i)).read() for i in xrange(790, 812) ]
y_test = [i.strip() for i in open('DATA/DP/test_set_lables.txt').readlines()]

print "Making a collection"
collection = make_collection(X_train, n)
regCollection = [re.compile(i) for i in collection]

# making vectors #
print "Making vectors"
vectorsTrain = make_vectors(X_train)
vectorsTest = make_vectors(X_test)

# Creating model #
print "Creating model"
model = SVC(kernel='precomputed')
   
# loading kernel + fiting model #
print "Creating kernel"
kernelTrain = []
for vector in vectorsTrain:
    t = []
    for vector1 in vectorsTrain:
        t.append(np.dot(vector, vector1))
    kernelTrain.append(t)
    
print "Fitting model"
model.fit(kernelTrain, y_train)

# loading train set + predicting labels #
print "Creating test set"
kernelTest = []
for vector in vectorsTest:
    t = []
    for vector1 in vectorsTrain:
        t.append(np.dot(vector, vector1))
    kernelTest.append(t)
    
print "Predicting"
y_predicted = model.predict(kernelTest)

# score #
print "################ RESULT ##################"
n = 0
for i in range(len(y_predicted)):
    if y_test[i] == y_predicted[i]:
        n += 1
        
print n, len(y_predicted)
