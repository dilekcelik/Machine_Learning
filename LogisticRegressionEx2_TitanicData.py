#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 15:54:16 2018
@author: dilekcelik
"""
import pandas as pd
import numpy as np

#Read Data
data = pd.read_excel("/Users/dilekcelik/Desktop/Machine_Learning/titanic.xls")
data.head()
#print(data.head())

#“body”, “name” ve “home.dest” columns are removed
data.drop(['body','name','home.dest'], 1, inplace=True)
data.head()
#print(data.head())

#To get statistical results, “describe” is used
#such as mean, std, min, max...
data.describe()
#print(data.describe())

#“age” ve “fare” colons counts value seems NaN (Not a Number)
#To correct this, we are calculating  median values for “age” ve “fare” columns
#Then, we replace the NaN values with Median Values
data["age"].fillna(data["age"].median(), inplace=True)
data["fare"].fillna(data["fare"].median(), inplace=True)
#data.describe()
print(data.describe())


#Visualising the Data
import matplotlib.pyplot as plt
colors = ["r","b","k"]
survived_ = data[data["survived"]== 1]["sex"].value_counts()
dead_ = data[data["survived"]==0]["sex"].value_counts()
data_ = pd.DataFrame([survived_, dead_])
data_.index = ["survived","dead"]
data_.plot.bar(stacked=True,color=colors)


fig = plt.figure(figsize=(15,10))
plt.hist([data[data["survived"]==1]["age"], data[data["survived"]==0]["age"]],histtype='bar',stacked=True,bins=20,color=["r","b"],width=3, label=["Survived","Dead"])
plt.xlabel("Age")
plt.ylabel("N")
plt.legend()

#probability of staying alive
import seaborn as sns
g = sns.factorplot(x="pclass", y="survived", hue="sex", data=data, size=6, kind="bar")
g.despine(left=True)
g.set_ylabels("Hayatta kalma olasılığı")
plt.show()









