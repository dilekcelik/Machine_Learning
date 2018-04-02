#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 15:47:00 2018
@author: dilekcelik

LINEAR REGRESSION
"""

import numpy as np # mathmatical calculations
import pandas as pd # data
from sklearn.linear_model import LinearRegression as lr
import matplotlib.pyplot as plt #for visualisations

#read data with pandas
data = pd.read_csv("/Users/dilekcelik/Desktop/Machine_Learning/linear.csv")

x = data["metrekare"]
y = data["fiyat"]
print(data)


x = x.reshape(99,1) #99X1 lik matrisler oldugunu belirttik
y = y.reshape(99,1)
print(data)

lineerregresyon = lr() #Define lineer regression
lineerregresyon.fit(x,y) # fit x and y, x ve y'yi oturttuk
lineerregresyon.predict(x)
#dogrumuzun formulu = mx+b, m= egim, b = kesistigi nokta

#print('Egim (m): /n', lineerregresyon.coef_)
#print('Y de kesistigi yer: ', lineerregresyon.intercept_)

m = lineerregresyon.coef_
b = lineerregresyon.intercept_
#print("Denklem: ")
#print("y=",m,"x+",b)

#SIMDI NASIL GOZUKTUGUNE BAKALIM, CIZDIRECEGIZ
a = np.arange(150)
a=a.reshape(150,1) #2D Matrix yapmam gerekti, burasi yokken hata verdi

plt.scatter(x,y)
plt.plot(a,m*a+b)
plt.show()