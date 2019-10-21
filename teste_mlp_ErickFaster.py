# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 20:30:26 2019

@author: pamsb
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 21:03:28 2019

@author: pamsb
"""


 # import numpy and pandas, and DataFrame / Series

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
import scipy.sparse as sparse


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

df_val.reset_index(drop=True, inplace=True)

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

df_train=df_train.values
df_val=df_val.values

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), i:(i+look_back)]
        dataX.append(a)
        dataY.append(dataset[ i + look_back, i + look_back])
    return np.array(dataX), np.array(dataY)
# reshape into X=t and Y=t+1
look_back = 1
X_train, y_train = create_dataset(df_train, look_back)
X_val, y_val = create_dataset(df_val, look_back)

#arr = np.ones(X_train.shape)
#X_train_novo=np.vstack((arr, X_train))

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

#X_train, y_train = makeXy(df_train['scaled_Vazao'],1)
#print('Shape of train arrays:', X_train.shape, y_train.shape)
#
#
#X_val, y_val = makeXy(df_val['scaled_Vazao'],1)
#X_train = df_train['scaled_Vazao'].shift(4, fill_value=0)
#X_train=np.array(X_train)
#y_train = df_train['scaled_Vazao']
#y_train=np.array(y_train)
print('Shape of train arrays:', X_train.shape, y_train.shape)
#X_val = df_val['scaled_Vazao'].shift(4, fill_value=0)
#X_val=np.array(X_val)
#y_val = df_val['scaled_Vazao']
#y_val=np.array(y_val)
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

'''GRID SEARCH'''



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
y_train = scaler_y_train.inverse_transform(y_train)
y_val = scaler_y_val.inverse_transform(y_val)

previsoes_train = scaler_y_train.inverse_transform(previsoes_train)
previsoes_test = scaler_y_val.inverse_transform(previsoes_test)

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



