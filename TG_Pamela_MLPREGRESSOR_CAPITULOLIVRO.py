# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 21:03:28 2019

@author: pamsb
"""


 # import numpy and pandas, and DataFrame / Series
from __future__ import print_function
import os
import sys
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import datetime

import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
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


matriz_indices_vazoes = pd.read_csv('C:\\Users\\pamsb\\OneDrive\\Documentos\\TG Python\\Matriz_vazao_regress.csv', sep=';')


matriz_indices_vazoes.head()


df = pd.DataFrame(matriz_indices_vazoes, columns = ['Ano','Mes', 'vazao'])

# Get column names first

# Create the Scaler object
df['datetime'] = df[['Ano', 'Mes']].apply(lambda row: datetime.datetime(year=row['Ano'],month=row['Mes'],  day=1), axis=1)
df.sort_values('datetime', ascending=True, inplace=True)
g = sns.tsplot(df['vazao'])
g.set_title('Time series of Vazão')
g.set_xlabel('Index')
g.set_ylabel('Vazão')
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
df['scaled_Vazao'] = scaler.fit_transform(np.array(df['vazao']).reshape(-1, 1))
split_date = datetime.datetime(year=2004, month=1, day=1)
df_train = df.loc[df['datetime']<split_date]
df_val = df.loc[df['datetime']>=split_date]
print('Shape of train:', df_train.shape)
print('Shape of test:', df_val.shape)



plt.figure(figsize=(5.5, 5.5))
g = sns.tsplot(df_train['scaled_Vazao'], color='b')
g.set_title('Time series of scaled Vazão in train set')
g.set_xlabel('Index')
g.set_ylabel('Scaled Vazão readings')
plt.figure(figsize=(5.5, 5.5))
g = sns.tsplot(df_val['scaled_Vazao'], color='r')
g.set_title('Time series of standardized Vazão in validation set')
g.set_xlabel('Index')
g.set_ylabel('Standardized Vazão readings')




def makeXy(ts, nb_timesteps):
    """
    Input: 
           ts: original time series
           nb_timesteps: number of time steps in the regressors
    Output: 
           X: 2-D array of regressors
           y: 1-D array of target 
    """
    X = []
    y = []
    for i in range(nb_timesteps, ts.shape[0]):
        X.append(list(ts.loc[i-nb_timesteps:i-1]))
        y.append(ts.loc[i])
    X, y = np.array(X), np.array(y)
    return X, y

X_train, y_train = makeXy(df_train['scaled_Vazao'],2)
print('Shape of train arrays:', X_train.shape, y_train.shape)

X_val, y_val = makeXy(df_val['scaled_Vazao'],2)
print('Shape of validation arrays:', X_val.shape, y_val.shape)





from keras.layers import Dense, Input, Dropout
from keras.optimizers import SGD
from keras.models import Model
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
input_layer = Input(shape=(1,), dtype='float32')

dense1 = Dense(32, activation='linear')(input_layer)
dense2 = Dense(16, activation='linear')(dense1)
dense3 = Dense(16, activation='linear')(dense2)
dropout_layer = Dropout(0.2)(dense3)
output_layer = Dense(1, activation='linear')(dropout_layer)
ts_model = Model(inputs=input_layer, outputs=output_layer)
ts_model.compile(loss='mean_squared_error', optimizer='adam')
ts_model.summary()

completeName='PRSA_data_VAZ_MLP_weights.{epoch:02d}-{val_loss:.4f}.hdf5'
save_weights_at=os.path.join(os.path.expanduser('~'),'Documents',completeName)

save_best = ModelCheckpoint(save_weights_at, monitor='val_loss', verbose=0,
 save_best_only=True, save_weights_only=False,
mode='min',
 period=1)
ts_model.fit(x=X_train, y=y_train, batch_size=16, epochs=20,
 verbose=1, callbacks=[save_best], validation_data=(X_val,
y_val),
 shuffle=True)


completeName='PRSA_data_VAZ_MLP_weights.11-0.0001.hdf5'
best_model = load_model(os.path.join(os.path.expanduser('~'),'Documents',completeName))

preds = best_model.predict(X_val)
pred_VAZ = scaler.inverse_transform(preds)
pred_VAZ = np.squeeze(pred_VAZ)


from sklearn.metrics import r2_score

r2 = r2_score(df_val['vazao'].loc[7:], pred_VAZ)
print('R-squared for the validation set:', round(r2,4))

#Let's plot the first 50 actual and predicted values of air pressure.
plt.figure(figsize=(5.5, 5.5))
plt.plot(range(50), df_val['vazao'].loc[7:56], linestyle='-', marker='*', color='r')
plt.plot(range(50), pred_VAZ[:50], linestyle='-', marker='.', color='b')
plt.legend(['Actual','Predicted'], loc=2)
plt.title('Actual vs Predicted Vazão')
plt.ylabel('Vazão')
plt.xlabel('Index')
 
 
    


 


