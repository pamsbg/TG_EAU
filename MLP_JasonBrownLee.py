# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 21:41:07 2019

@author: pamsb
"""

import numpy
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense

import os
import sys
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import datetime

import pandas as pd
from pandas import DataFrame, Series

from sklearn import datasets, linear_model, preprocessing
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score
from mpl_toolkits.mplot3d import Axes3D
from prettytable import PrettyTable
from prettytable import from_csv
from prettytable import from_html
from sklearn.inspection import partial_dependence
from sklearn.inspection import plot_partial_dependence
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset
matriz_indices_vazoes = pd.read_csv('C:\\Users\\pamsb\\OneDrive\\Documentos\\TG Python\\Matriz_vazao_regress.csv', sep=';')
#dataframe = pandas.read_csv( international-airline-passengers.csv , usecols=[1],
#engine= python , skipfooter=3)
dataframe = pd.DataFrame(matriz_indices_vazoes, columns = ['Ano','Mes', 'vazao'])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
#escalonar valores
dataframe['scaled_Vazao'] = scaler.fit_transform(np.array(dataframe['vazao']).reshape(-1, 1))
#ordenar valores por data para poder dividir
dataframe['datetime'] = dataframe[['Ano', 'Mes']].apply(lambda row: datetime.datetime(year=row['Ano'],month=row['Mes'],  day=1), axis=1)
dataframe.sort_values('datetime', ascending=True, inplace=True)
#data de corte
split_date = datetime.datetime(year=2004, month=1, day=1)
#dataset de treino
df_train = dataframe.loc[dataframe['datetime']<split_date]
#dataset de validação
df_val = dataframe.loc[dataframe['datetime']>=split_date]

#resetando index
df_val.reset_index(drop=True, inplace=True)
#criar lags
X_train = df_train['scaled_Vazao'].shift(1, fill_value=0)
X_train=np.array(X_train)
y_train = df_train['scaled_Vazao']
y_train=np.array(y_train)
print('Shape of train arrays:', X_train.shape, y_train.shape)
#criar lags
X_val = df_val['scaled_Vazao'].shift(1, fill_value=0)
X_val=np.array(X_val)
y_val = df_val['scaled_Vazao']
y_val=np.array(y_val)
print('Shape of validation arrays:', X_val.shape, y_val.shape)


from sklearn.preprocessing import StandardScaler
scaler_x_train = StandardScaler()
X_train=X_train.reshape(-1,1)
X_train = scaler_x_train.fit_transform(np.array(X_train))
#X_train = scaler_x_train.fit_transform(X_train).values.reshape(-1,1)
X_val=X_val.reshape(-1,1)
scaler_x_test = StandardScaler()
X_val = scaler_x_test.fit_transform(np.array(X_val))
#X_val = scaler_x_test.fit_transform(X_val).values.reshape(-1,1)

#Necessario Escalonar
scaler_y_train = StandardScaler()
y_train=y_train.reshape(-1,1)
y_train = scaler_y_train.fit_transform(y_train)
y_val=y_val.reshape(-1,1)
scaler_y_val = StandardScaler()
y_val = scaler_y_val.fit_transform(y_val)
dataset=dataframe.values

# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))
# convert an array of values into a dataset t matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)
# reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
# create and fit Multilayer Perceptron model
model = Sequential()
model.add(Dense(8, input_dim=look_back, activation= 'tanh' ))
model.add(Dense(1))
model.compile(loss= 'mean_squared_error' , optimizer= 'adam' )
model.fit(trainX, trainY, nb_epoch=200, batch_size=2, verbose=2)
# Estimate model performance
trainScore = model.evaluate(trainX, trainY, verbose=0)

testScore = model.evaluate(testX, testY, verbose=0)

# generate predictions for training
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
plt.plot(dataset)
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()