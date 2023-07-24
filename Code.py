# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 13:49:59 2021

@author: Swaroop Honrao
"""

#Import Library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Import dataset
data = pd.read_csv('train.csv')
x = data.iloc[:, [4]]
y = data.iloc[:, 1]

#Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [4])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

#splitting dataset into train set and test set
from sklearn.preprocessing import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)

#Linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#Predicting result
y_pred = regressor.predict(x_test)

# Visualising the Linear Regression results
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Titanic survival (Linear Regression)')
plt.xlabel('gender')
plt.ylabel('Survived')
plt.show()

# Visualising the Linear Regression results
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_test, regressor.predict(x_test), color = 'blue')
plt.title('Titanic (Linear Regression)')
plt.xlabel('gender')
plt.ylabel('Survived')
plt.show()