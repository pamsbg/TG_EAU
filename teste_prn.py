# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 21:14:03 2019

@author: pamsb
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 11:45:27 2019

@author: pamsb
"""



import os
import sys
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns
import datetime


#Read the dataset into a pandas.DataFrame
df = pd.read_csv('C:\\Users\\pamsb\\OneDrive\\Documentos\\TG Python\\Vazões_Mensais_1979_2014_ptsprincipais.prn', delimiter= "\t")

print('Shape of the dataframe:', df.shape)

df['datetime'] = df[['Ano', 'Mes']].apply(lambda row: datetime.datetime(year=row['Ano'], month=row['Mes'], day=1
                                                                                          ), axis=1)
df.sort_values('datetime', ascending=True, inplace=True)

#Let us draw a box plot to visualize the central tendency and dispersion of PRES
plt.figure(figsize=(5.5, 5.5))
g = sns.boxplot(df['Camargos'])
g.set_title('Box plot of Air Pressure')

plt.figure(figsize=(5.5, 5.5))
g = sns.tsplot(df['Camargos'])
g.set_title('Time series of Air Pressure')
g.set_xlabel('Index')
g.set_ylabel('Air Pressure readings in hPa')


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
df['scaled_vazao'] = scaler.fit_transform(np.array(df['Camargos']).reshape(-1, 1))

split_date = datetime.datetime(year=2004, month=1, day=1)
df_train = df.loc[df['datetime']<split_date]
df_val = df.loc[df['datetime']>=split_date]
print('Shape of train:', df_train.shape)
print('Shape of test:', df_val.shape)

df_val.reset_index(drop=True, inplace=True)


plt.figure(figsize=(5.5, 5.5))
g = sns.tsplot(df_train['scaled_vazao'], color='b')
g.set_title('Time series of scaled stream flow in train set')
g.set_xlabel('Index')
g.set_ylabel('Scaled stream flow readings')


plt.figure(figsize=(5.5, 5.5))
g = sns.tsplot(df_val['scaled_vazao'], color='r')
g.set_title('Time series of scaled stream flow in validation set')
g.set_xlabel('Index')
g.set_ylabel('Scaled Air Pressure readings')

def my_r2(y, yhat):
    r2 = np.sum(np.power((np.subtract(yhat, np.mean(y))),2))/np.sum(np.power((np.subtract(y, np.mean(y))),2))
    
    return r2


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

X_train, y_train = makeXy(df_train['scaled_vazao'], 7)
print('Shape of train arrays:', X_train.shape, y_train.shape)

X_val, y_val = makeXy(df_val['scaled_vazao'], 7)
print('Shape of validation arrays:', X_val.shape, y_val.shape)


from keras.layers import Dense, Input, Dropout
from keras.optimizers import SGD
from keras.models import Model
from keras.models import load_model
from keras.callbacks import ModelCheckpoint


input_layer = Input(shape=(7,), dtype='float32')


dense1 = Dense(32, activation='linear')(input_layer)
dense2 = Dense(16, activation='linear')(dense1)
dense3 = Dense(16, activation='linear')(dense2)

dropout_layer = Dropout(0.2)(dense3)

output_layer = Dense(1, activation='linear')(dropout_layer)


ts_model = Model(inputs=input_layer, outputs=output_layer)
ts_model.compile(loss='mean_squared_error', optimizer='adam')
ts_model.summary()
completeName='PRSA_data_vazao_MLP_weights.{epoch:02d}-{val_loss:.4f}.hdf5'
save_weights_at=os.path.join(os.path.expanduser('~'),'Documents',completeName)
#save_weights_at = os.path.join('keras_models', 'PRSA_data_Air_Pressure_MLP_weights.{epoch:02d}-{val_loss:.4f}.hdf5')
save_best = ModelCheckpoint(save_weights_at, monitor='val_loss', verbose=0,
                            save_best_only=True, save_weights_only=False, mode='min',
                            period=1)
ts_model.fit(x=X_train, y=y_train, batch_size=16, epochs=20,
             verbose=1, callbacks=[save_best], validation_data=(X_val, y_val),
             shuffle=True)

completeName='PRSA_data_vazao_MLP_weights.11-0.0129.hdf5'
best_model = load_model(os.path.join(os.path.expanduser('~'),'Documents',completeName))
preds = best_model.predict(X_val)
pred_PRES = scaler.inverse_transform(preds)
pred_PRES = np.squeeze(pred_PRES)


from sklearn.metrics import r2_score
r2 = r2_score(df_val['Camargos'].loc[7:], pred_PRES)
#r2 = my_r2(y_val, pred_PRES)
print('R-squared for the validation set:', round(r2,4))

#Let's plot the first 50 actual and predicted values of air pressure.
plt.figure(figsize=(5.5, 5.5))
plt.plot(range(50), df_val['Camargos'].loc[7:56], linestyle='-', marker='*', color='r')
plt.plot(range(50), pred_PRES[:50], linestyle='-', marker='.', color='b')
plt.legend(['Actual','Predicted'], loc=2)
plt.title('Actual vs Predicted Stream Flow')
plt.ylabel('Stream Flow')
plt.xlabel('Index')


