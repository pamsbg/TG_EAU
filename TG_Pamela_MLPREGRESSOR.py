# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 10:15:54 2019

@author: pamsb
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 20:37:37 2019

@author: pamsb
"""

 # import numpy and pandas, and DataFrame / Series
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, preprocessing
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import seaborn as sns
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

#miv = matriz_indices_vazoes
miv = pd.DataFrame(matriz_indices_vazoes, columns = ['Ano','Mes','AMO','NAO','AMM','TSA','TNA','PDO','MEI','EMI','AAO','AO','PNA','SOLAR','QBO', 'vazao'])
normalized_miv = preprocessing.normalize(miv)
# Get column names first
names = miv.columns
# Create the Scaler object
scaler = preprocessing.StandardScaler()
# Fit your data on the scaler object
scaled_miv = scaler.fit_transform(miv)
scaled_miv = pd.DataFrame(scaled_miv, columns=names)
#miv_y=miv.loc[:,'vazao']
# Use only one feature
#miv_X = miv.loc(axis=0)[:,'Ano','Mês','AMO','NAO','AMM','TSA','TNA','PDO','MEI','EMI','AAO','AO','PNA','SOLAR','QBO']
miv_X=scaled_miv[['Ano','Mes','AMO','NAO','AMM','TSA','TNA','PDO','MEI','EMI','AAO','AO','PNA','SOLAR','QBO']]
miv_y = scaled_miv[['vazao']]
#
#miv_X_df=pd.DataFrame(data=miv_X[1:,1:],    # values
#...              index=miv_X[1:,0],    # 1st column as index
#...              columns=miv_X[0,1:])
#miv_y_df=pd.DataFrame(data=miv_y[1:,1:],    # values
#...              index=miv_y[1:,0],    # 1st column as index
#...              columns=miv_y[0,1:])
# Split the data into training/testing sets
#miv_X_train = miv_X[:311].values.reshape(-1,1)
#miv_X_test = miv_X[312:].values.reshape(-1,1)


miv_X_train = miv_X[:311]
miv_X_test = miv_X[312:]
# Split the targets into training/testing sets
miv_y_train = miv_y[:311].values.reshape(-1,1)
miv_y_test = miv_y[312:].values.reshape(-1,1)


# Create linear regression object
f=('identity', 'logistic', 'tanh', 'relu')
contador =0

    # Cria a tabela
x = PrettyTable(["ID","Método", "Activation", "r", "R²", "RMSE", 'MAE'])

#for i in f:

regr = MLPRegressor(activation='tanh', hidden_layer_sizes=(15,15,15), max_iter=1000, solver='lbfgs', tol='1e-5')



    # Train the model using the training sets
regr.fit(miv_X_train, miv_y_train)

    # Make predictions using the testing set
miv_y_pred = regr.predict(miv_X_test)
miv_y_pred_out = regr.predict(miv_X_test)
miv_y_pred_in = regr.predict(miv_X_train)

#    regr.coefs_
    # The coefficients
#    print('Coefficients: \n', regr.coefs_)
    # The mean squared error
print("Mean squared error: %.2f"  % mean_squared_error(miv_y_test, miv_y_pred))
    # Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(miv_y_test, miv_y_pred))
print('MAE score: %.2f' % mean_absolute_error(miv_y_test, miv_y_pred))



a = [-5, 0, 5]
b = [-5, 0, 5]
    # Plot outputs
plt.scatter(miv_y_test, miv_y_pred, color='black')
plt.axis((-5,5,-5,5))
plt.plot(a,b)
#plt.plot(miv_y_test, miv_y_pred, color='blue', linewidth=1)

plt.xticks(())
plt.yticks(())

plt.show()


plt.scatter(miv_y_train, miv_y_pred_in, color='black')
plt.axis((-5,5,-5,5))
plt.plot(a,b)
#plt.plot(miv_y_test, miv_y_pred, color='blue', linewidth=1)

plt.xticks(())
plt.yticks(())

plt.show()




    # Alinha as colunas
    #x.align["ID"] = "l"
    #x.align["Método"] = "l"
    #x.align["Activation"] = "r"
    #x.align["r"] = "r"
    #x.align["R²"] = "r"
    #x.align["RMSE"] = "r"
    #x.align["MAE"] = "r"


    # Deixa um espaço entre a borda das colunas e o conteúdo (default)
    #x.padding_width = 1
#    contador = contador + 1
#    x.add_row([contador,"MLP",i, miv_y_pred.corr(miv_y_train['vazao']), r2_score(miv_y_test, miv_y_pred),mean_squared_error(miv_y_test, miv_y_pred), mean_absolute_error(miv_y_test, miv_y_pred) ])
#    print(x)

