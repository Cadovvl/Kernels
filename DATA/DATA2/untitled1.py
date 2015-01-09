# -*- coding: utf-8 -*-
"""
Created on Wed Dec 24 22:18:50 2014

@author: olga
"""


import os

path="./original+info/pos"

for root, dirs, files in os.walk(path):
    n=0
    for name in files:
        f=open(os.path.join(root,name)).read()
        if n<800:
            w=open('80-20/TrainSet/train{0}.txt'.format(n+800), 'w').write(f)
        else:
            w=open('80-20/TestSet/test{0}.txt'.format(n-600),'w').write(f)
        n+=1  