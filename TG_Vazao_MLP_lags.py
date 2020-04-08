# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 20:19:08 2020

@author: pamsb
"""


#redes neurais artificiais

import numpy as np
import matplotlib.pylab as plt
#matplotlib.pylab inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize']=15,6
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import math
dateparse=lambda dates: pd.datetime.strptime(dates, '%Y-%m')
df = pd.read_csv('Matriz_vazao_regress.csv', sep=';', parse_dates=['Month'], index_col='Month', date_parser =dateparse)

dataprep=df.iloc[:,13:14]
data=dataprep.values

np.random.seed(3)
data = data.astype('float32')
scaler = MinMaxScaler(feature_range=(0,1))
data = scaler.fit_transform(data)
train = data[:300,:]
test = data[300:,:]

def prepare_data(data, lags):
    X,y = [],[]
    for row in range(len(data)-lags-1):
        a = data[row:(row+lags),0]
        X.append(a)
        y.append(data[row-lags,0])
    return np.array(X),np.array(y)
                  

    
lags =12
X_train,y_train = prepare_data(train,lags)
X_test,y_test = prepare_data(test,lags)
y_true = y_test

plt.plot(y_test, label='Dados Originais de Vazão | y ou t+1', color ='blue')    
plt.plot(X_test, label='Dados Passados | X ou t', color='orange')
plt.legend(loc='upper left')
plt.title('Dados passados em um período')
plt.show()

X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1]))


mdl = Sequential()
mdl.add(Dense(12,input_dim=lags, activation='softplus'))
mdl.add(Dense(1))
mdl.compile(loss='mean_squared_error', optimizer='adam')
mdl.fit(X_train, y_train, epochs=2000, batch_size=10,verbose=0)

# train_score=mdl.evaluate(X_train, y_train, verbose=0)
#print('Pontuação de Treino: ' + train_score + 'MSE' + math.sqrt(train_score)+' RMSE')

# test_score=mdl.evaluate(X_test, y_test, verbose=0)
#print('Pontuação de Treino: {:,2f} MSE ({:,2f} RMSE)'.format(test_score, math.sqrt(test_score)))


train_predict = mdl.predict(X_train)
test_predict = mdl.predict(X_test)


mlp_r2=r2_score(y_test,test_predict)
mlp_rmse = mean_squared_error(y_test, test_predict)
mlp_mae = mean_absolute_error(y_test, test_predict)




train_predict_plot =np.empty_like(data)
train_predict_plot[:,:]=np.nan
train_predict_plot[lags:len(train_predict)+lags,:]=train_predict


test_predict_plot =np.empty_like(data)
test_predict_plot[:,:]=np.nan
test_predict_plot[len(train_predict)+(lags*2)+1:len(data)-1,:]=test_predict

plt.plot(data, label='Observado', color='blue')
plt.plot(train_predict_plot, label='Previsão para os dados de treino', color='red', alpha=0.5)
plt.plot(test_predict_plot, label='Previsão para os dados de teste', color='yellow')
plt.legend(loc='best')
plt.show

plt.savefig('Grafico MLP Lag' + str(lags) +'.png')

from sklearn.linear_model import LinearRegression
rl = LinearRegression().fit(X_train,y_train)
rl_trainscore =rl.score(X_train, y_train)
rl_testscore=rl.score(X_test, y_test)
rl_predicttest =rl.predict(X_test)
rl_predicttrain =rl.predict(X_train)
rl_r2=r2_score(y_test,rl_predicttest)
rl_rmse = mean_squared_error(y_test, rl_predicttest)
rl_mae = mean_absolute_error(y_test, rl_predicttest)




rl_train_predict_plot =np.empty_like(data)
rl_train_predict_plot[:,:]=np.nan
rl_train_predict_plot[lags:len(rl_predicttrain)+lags,:]=rl_predicttrain


rl_test_predict_plot =np.empty_like(data)
rl_test_predict_plot[:,:]=np.nan
rl_test_predict_plot[len(rl_predicttest)+(lags*2)+1:len(data)-1,:]=rl_predicttest

plt.plot(data, label='Observado', color='blue')
plt.plot(rl_train_predict_plot, label='Previsão para os dados de treino', color='red', alpha=0.5)
plt.plot(rl_test_predict_plot, label='Previsão para os dados de teste', color='yellow')
plt.legend(loc='best')
plt.show

plt.savefig('Grafico OLS Lag' + str(lags) +'.png')

