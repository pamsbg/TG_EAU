# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 21:34:57 2019

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
from sklearn import datasets, linear_model, preprocessing
from matplotlib import pyplot as plt
import seaborn as sns
import datetime


from matplotlib import pyplot as plt
import seaborn as sns
import datetime


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
import scipy.sparse as sparse

#Read the dataset into a pandas.DataFrame
df = pd.read_csv('C:\\Users\\pamsb\\OneDrive\\Documentos\\TG Python\\Matriz_vazao_regress.csv', sep=';')

print('Shape of the dataframe:', df.shape)

df['datetime'] = df[['Ano', 'Mes']].apply(lambda row: datetime.datetime(year=row['Ano'], month=row['Mes'], day=1
                                                                                          ), axis=1)
df.sort_values('datetime', ascending=True, inplace=True)

#Let us draw a box plot to visualize the central tendency and dispersion of PRES
plt.figure(figsize=(5.5, 5.5))
g = sns.boxplot(df['vazao'])
g.set_title('Box plot of Air Pressure')

plt.figure(figsize=(5.5, 5.5))
g = sns.tsplot(df['vazao'])
g.set_title('Time series of Air Pressure')
g.set_xlabel('Index')
g.set_ylabel('Air Pressure readings in hPa')


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
df['scaled_vazao'] = scaler.fit_transform(np.array(df['vazao']).reshape(-1, 1))

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

vetor=np.arange(1,20)
vetor = pd.DataFrame(vetor)
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


def lagxy(vetor,lag):
    X=[]
    y=[]
    for i in range(lag+1):
        np.hstack(X, )
    return X,y


x_vetor, y_vetor = makeXy(vetor,1)


X_train, y_train = makeXy(df_train['scaled_vazao'], 7)
print('Shape of train arrays:', X_train.shape, y_train.shape)

X_val, y_val = makeXy(df_val['scaled_vazao'], 7)
print('Shape of validation arrays:', X_val.shape, y_val.shape)


#Dados Fixos
mlp = MLPRegressor(max_iter=1000, solver = 'adam', activation = 'tanh', 
                   alpha = 0.0001, tol=0.00001)

#Dados Variaveis
parameter_space = {
    'hidden_layer_sizes': [(1,1,), (1,2,), (1,3,), (1,4,), (1,5,), (1,6,),
                           (2,1,), (2,2,), (2,3,), (2,4,), (2,5,), (2,6,),
                           (3,1,), (3,2,), (3,3,), (3,4,), (3,5,), (3,6,),
                           (4,1,), (4,2,), (4,3,), (4,4,), (4,5,), (4,6,),
                           (5,1,), (5,2,), (5,3,), (5,4,), (5,5,), (5,6,),
                           (6,1,), (6,2,), (6,3,), (6,4,), (6,5,), (6,6,),
                           (1,), (2,)]
}

#Grid Search
from sklearn.model_selection import GridSearchCV
regressor = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)
regressor.fit(X_train, y_train)


#Resultados

medias = regressor.cv_results_['mean_test_score']
desvios = regressor.cv_results_['std_test_score']
ranks = regressor.cv_results_['rank_test_score']
for media, desvio, rank, params in zip(medias, desvios, ranks, regressor.cv_results_['params']):
    print("Parametros: %r \nRank:%d, %0.3f (+-%0.03f)\n" %(params, rank, media, desvio))

#Print Melhor Parametro
print("\n***********************")
print('Melhor Parametro:\n', regressor.best_params_)
print('Score: ',regressor.best_score_)
print('Index: ',regressor.best_index_)
print('Estimator: ',regressor.best_estimator_)
print("***********************\n")

#Assegurar de que a melhor configuracao seja captada
regressor = regressor.best_estimator_

#Calculo do Score - Deve ocorrer antes do desescalonamento
score_train = regressor.score(X_train, y_train)
score_test = regressor.score(X_val, y_val)

previsoes_train = regressor.predict(X_train)
previsoes_test = regressor.predict(X_val)

#Necessario Desescalonar
y_train = scaler.inverse_transform(y_train).reshape(1,-1)
y_val = scaler.inverse_transform(y_val).reshape(1,-1)

previsoes_train = scaler.inverse_transform(previsoes_train).reshape(1,-1)
previsoes_test = scaler.inverse_transform(previsoes_test).reshape(1,-1)

'''
Tratamento dos Dados
'''

from math import sqrt

#Dados estatisticos do treinamento
mae_train = mean_absolute_error(y_train, previsoes_train)
mse_train = mean_squared_error(y_train, previsoes_train)
rmse_train = sqrt(mean_squared_error(y_train, previsoes_train))

#Dados estatisticos dos testes
mae_test = mean_absolute_error(y_val, previsoes_test)
mse_test = mean_squared_error(y_val, previsoes_test)
rmse_test = sqrt(mean_squared_error(y_val, previsoes_test))
r2_test = r2_score(y_val, previsoes_test)
import matplotlib.pyplot as plt

plt.title('Valor real x Valor previsto')
plt.xlabel('Valores reais')
plt.ylabel('Valores previstos')

plt.scatter(y_val, previsoes_test, color='black')
plt.axis((0,5,0,5))
plt.plot([0,5],[0,5])
plt.savefig('grafico.jpeg')

plt.show()


