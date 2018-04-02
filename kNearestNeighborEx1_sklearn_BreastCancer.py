#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 17:26:29 2018
@author: dilekcelik
"""
import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?',-99999,inplace=True)

#Delete Useless Data such as ID
df.drop(['id'],1,inplace=True)

#Define x and y
#x includes everything but not class
x = np.array(df.drop(['class'],1))
#y includes only class
y = np.array(df['class'])

#Split the data as test and train
x_train, x_test, y_train, y_test = cross_validation.train_test_split(x,y,test_size=0.2)

#define k, and fit the data
clf = KNeighborsClassifier(n_neighbors = 9)
clf.fit(x_train, y_train)

accuracy = clf.score(x_test, y_test)
print(accuracy)

#Example Test Data
example_measures = np.array([[4,2,1,1,1,2,3,2,1]])
#Predictions
prediction = clf.predict(example_measures)
#this tells us whether the cancer or not
print(prediction)



