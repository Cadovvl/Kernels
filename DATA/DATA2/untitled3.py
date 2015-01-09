# -*- coding: utf-8 -*-
"""
Created on Wed Dec 24 22:42:38 2014

@author: olga
"""

text=''
for i in range(200):
    text=text+'0'+'\n'
for i in range(200):
    text=text+'1'+'\n'
    
w=open('80-20/test_set_lables.txt', 'w').write(text[:-1])