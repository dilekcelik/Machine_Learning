#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 15:47:00 2018
@author: dilekcelik

LINEAR REGRESSION
"""

import numpy as np # mathmatical calculations
import pandas as pd # data
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt #for visualisations

#read data
data = pd.read_csv("/Users/dilekcelik/Desktop/Machine_Learning/linear.csv")
print(data) #print data

x = data["metrekare"]
y = data["fiyat"]
x = pd.DataFrame.as_matrix(x) #Turn it into Pandas Matrix
y = pd.DataFrame.as_matrix(y) 

print(x)
print(y) #see what we have done

plt.scatter(x,y) #To see what we have done in 2D graph

#Polyfit - Numpy bizim icin grafige oturtuyor cizgimizi
m,b = np.polyfit(x,y,1)
a = np.arange(150) #Denklem hzir, a'nin araligini ayarlayalim

plt.scatter(x,y) #Scatter ile nokt cizimleri yapiyoruz
plt.plot(m*a+b) #cizgiyi cizdik


#TEST EDELIM:
MeterSqure = int(input("Kac metrekare? :"))
Guess = m*MeterSqure+b
print(Guess)
plt.scatter(MeterSqure, Guess, c="pink", marker=">")
plt.show()
print("y=",m,"x+",b)
