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
#g = sns.tsplot(df['vazao'])
#g.set_title('Time series of Vazão')
#g.set_xlabel('Index')
#g.set_ylabel('Vazão')
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
df['scaled_Vazao'] = scaler.fit_transform(np.array(df['vazao']).reshape(-1, 1))

split_date = datetime.datetime(year=2004, month=1, day=1)
df_train = df.loc[df['datetime']<split_date]
df_val = df.loc[df['datetime']>=split_date]

print('Shape of train:', df_train.shape)
print('Shape of test:', df_val.shape)

df_val.reset_index(drop=True, inplace=True)

#plt.figure(figsize=(5.5, 5.5))
#g = sns.tsplot(df_train['scaled_Vazao'], color='b')
#g.set_title('Time series of scaled Vazão in train set')
#g.set_xlabel('Index')
#g.set_ylabel('Scaled Vazão readings')
#plt.figure(figsize=(5.5, 5.5))
#g = sns.tsplot(df_val['scaled_Vazao'], color='r')
#g.set_title('Time series of standardized Vazão in validation set')
#g.set_xlabel('Index')
#g.set_ylabel('Standardized Vazão readings')


#
#
#def makeXy(ts, nb_timesteps):
#    """
#    Input: 
#           ts: original time series
#           nb_timesteps: number of time steps in the regressors
#    Output: 
#           X: 2-D array of regressors
#           y: 1-D array of target 
#    """
#    X = []
#    y = []
#    for i in range(nb_timesteps, ts.shape[0]):
#        X.append(list(ts.loc[i-nb_timesteps:i-1]))
#        y.append(ts.loc[i])
#    X, y = np.array(X), np.array(y)
#    return X, y
#
#X_train, y_train = makeXy(df_train['scaled_Vazao'],6)
#print('Shape of train arrays:', X_train.shape, y_train.shape)
#
#
#X_val, y_val = makeXy(df_val['scaled_Vazao'],6)
#print('Shape of validation arrays:', X_val.shape, y_val.shape)
#
X_train = df_train['scaled_Vazao'].shift(1, fill_value=0)
X_train=np.array(X_train)
y_train = df_train['scaled_Vazao']
y_train=np.array(y_train)
print('Shape of train arrays:', X_train.shape, y_train.shape)
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



from keras.layers import Dense, Input, Dropout
from keras.optimizers import SGD
from keras.models import Model
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
#input_layer = Input(shape=(1,), dtype='float32')
input_layer=Input(X_train.shape[1])

dense1 = Dense(32, activation='tanh')(input_layer)
dense2 = Dense(16, activation='tanh')(dense1)
dense3 = Dense(16, activation='tanh')(dense2)
dropout_layer = Dropout(0.2)(dense3)
output_layer = Dense(1, activation='tanh')(dropout_layer)
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


completeName='PRSA_data_VAZ_MLP_weights.20-0.0128.hdf5'
best_model = load_model(os.path.join(os.path.expanduser('~'),'Documents',completeName))

preds = best_model.predict(X_val)
pred_VAZ = scaler.inverse_transform(preds)
pred_VAZ = np.squeeze(pred_VAZ)




r2 = r2_score(df_val['vazao'], pred_VAZ)
print('R-squared for the validation set:', round(r2,4))

#Let's plot the first 50 actual and predicted values of air pressure.
plt.figure(figsize=(5.5, 5.5))
plt.plot(range(50), df_val['vazao'].loc[7:56], linestyle='-', marker='*', color='r')
plt.plot(range(50), pred_VAZ[:50], linestyle='-', marker='.', color='b')
plt.legend(['Actual','Predicted'], loc=2)
plt.title('Actual vs Predicted Vazão')
plt.ylabel('Vazão')
plt.xlabel('Index')

#regressão linear
# Create linear regression object
regr = linear_model.LinearRegression()
regr.get_params

# Train the model using the training sets
regr.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = regr.predict(X_val)
# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_val, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_val, y_pred))

# Plot outputs
plt.scatter(y_val, y_pred,  color='black')
a = [-5, 0, 5]
b = [-5, 0, 5]
plt.plot(a, b)

plt.xticks(())
plt.yticks(())

plt.show()

 
from matplotlib import pyplot
from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error




from pandas.plotting import autocorrelation_plot

autocorrelation_plot(df['scaled_Vazao'])
pyplot.show()
from statsmodels.graphics.tsaplots import plot_acf

plot_acf(df['scaled_Vazao'], lags=12)
pyplot.show()
# train autoregression
model = AR(X_train)
model_fit = model.fit()
window = model_fit.k_ar
coef = model_fit.params
# walk forward over time steps in test
history = X_train[len(X_train)-window:]
history = [history[i] for i in range(len(history))]
predictions = list()
for t in range(len(X_val)):
	length = len(history)
	lag = [history[i] for i in range(length-window,length)]
	yhat = coef[0]
	for d in range(window):
		yhat += coef[d+1] * lag[window-d-1]
	obs = X_val[t]
	predictions.append(yhat)
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(X_val, predictions)
print('Test MSE: %.3f' % error)
# plot
pyplot.plot(X_val)
pyplot.plot(predictions, color='red')
pyplot.show()
    


 


