# -*- coding: utf-8 -*-
"""
- NN

Uso do metodo MLPRegression para a previsao da rugosidade superficial
de materiais, seguindo o banco de dados obtido por Hossain et al.
"""

'''
listaMAE_train = []
listaMAE_test = []
listaMSE_train = []
listaMSE_test = []
listaRMSE_train = []
listaRMSE_test = []
listaSCORE_train = []
listaSCORE_test = []
'''

'''
Pre-Processamento
'''
import pandas as pd
base = pd.read_csv('C:\\Users\\pamsb\\OneDrive\\Documentos\\TG Python\\Matriz_vazao_regress.csv', sep=';')
#base = pd.read_excel('Hossain.xlsx', encoding = 'ISO-8859-1')

#Necessario Slide em y, pra não dar erro!!
split_date = datetime.datetime(year=2004, month=1, day=1)
df_train = df.loc[df['datetime']<split_date]
df_val = df.loc[df['datetime']>=split_date]
X_train = base.iloc[0:68, 1:6].values
y_train = base.iloc[0:68:,6:7].values #slide

X_test = base.iloc[68:84,1:6].values
y_test = base.iloc[68:84,6:7].values #slide

from sklearn.preprocessing import StandardScaler
scaler_x_train = StandardScaler()
X_train = scaler_x_train.fit_transform(X_train)
scaler_x_test = StandardScaler()
X_test = scaler_x_test.fit_transform(X_test)

#Necessario Escalonar
scaler_y_train = StandardScaler()
y_train = scaler_y_train.fit_transform(y_train)
scaler_y_test = StandardScaler()
y_test = scaler_y_test.fit_transform(y_test)

'''GRID SEARCH'''

from sklearn.neural_network import MLPRegressor

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

'''
Processamento
'''

'''
from sklearn.neural_network import MLPRegressor
regressor = MLPRegressor(hidden_layer_sizes = (2,),
                         activation = 'tanh',
                         solver = 'adam',
                         max_iter = 1000,
                         alpha = 0.0001)
regressor.fit(X_train, y_train)
'''

'''
Previsao
'''

#Assegurar de que a melhor configuracao seja captada
regressor = regressor.best_estimator_

#Calculo do Score - Deve ocorrer antes do desescalonamento
score_train = regressor.score(X_train, y_train)
score_test = regressor.score(X_test, y_test)

previsoes_train = regressor.predict(X_train)
previsoes_test = regressor.predict(X_test)

#Necessario Desescalonar
y_train = scaler_y_train.inverse_transform(y_train)
y_test = scaler_y_test.inverse_transform(y_test)

previsoes_train = scaler_y_train.inverse_transform(previsoes_train)
previsoes_test = scaler_y_test.inverse_transform(previsoes_test)

'''
Tratamento dos Dados
'''
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt

#Dados estatisticos do treinamento
mae_train = mean_absolute_error(y_train, previsoes_train)
mse_train = mean_squared_error(y_train, previsoes_train)
rmse_train = sqrt(mean_squared_error(y_train, previsoes_train))

#Dados estatisticos dos testes
mae_test = mean_absolute_error(y_test, previsoes_test)
mse_test = mean_squared_error(y_test, previsoes_test)
rmse_test = sqrt(mean_squared_error(y_test, previsoes_test))

import matplotlib.pyplot as plt

plt.title('Valor real x Valor previsto')
plt.xlabel('Valores reais')
plt.ylabel('Valores previstos')

plt.scatter(y_test, previsoes_test, color='black')
plt.axis((0,5,0,5))
plt.plot([0,5],[0,5])
plt.savefig('grafico.jpeg')

plt.show()






'''
Listas
'''

'''
listaMAE_train.append(mae_train)
listaMAE_test.append(mae_test)
listaMSE_train.append(mse_train)
listaMSE_test.append(mse_test)
listaRMSE_train.append(rmse_train)
listaRMSE_test.append(rmse_test)
listaSCORE_train.append(score_train)
listaSCORE_test.append(score_test)

import numpy as np

print("%.4f" %np.mean(listaMAE_train))
print("%.4f" %np.mean(listaMAE_test))
print("%.4f" %np.mean(listaMSE_train))
print("%.4f" %np.mean(listaMSE_test))
print("%.4f" %np.mean(listaRMSE_train))
print("%.4f" %np.mean(listaRMSE_test))
print("%.4f" %np.mean(listaSCORE_train))
print("%.4f" %np.mean(listaSCORE_test))
'''
