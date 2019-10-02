# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 21:03:32 2019

@author: pamsb
"""

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
from sklearn.svm import SVR
from sklearn import datasets, linear_model
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


#miv_y=miv.loc[:,'vazao']
# Use only one feature
#miv_X = miv.loc(axis=0)[:,'Ano','Mês','AMO','NAO','AMM','TSA','TNA','PDO','MEI','EMI','AAO','AO','PNA','SOLAR','QBO']
miv_X=miv[['Ano','Mes','AMO','NAO','AMM','TSA','TNA','PDO','MEI','EMI','AAO','AO','PNA','SOLAR','QBO']]
miv_y = miv[['vazao']]
# Split the data into training/testing sets
miv_X_train = miv_X[:311]
miv_X_test = miv_X[312:]

# Split the targets into training/testing sets
miv_y_train = miv_y[:311]
miv_y_test = miv_y[312:]


# Create linear regression object
f=('identity', 'logistic', 'tanh', 'relu')
contador =0

    # Cria a tabela
x = PrettyTable(["ID","Método", "Activation", "r", "R²", "RMSE", 'MAE'])




# #############################################################################
# Fit regression model
svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
svr_lin = SVR(kernel='linear', C=100, gamma='auto')
svr_poly = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1,
               coef0=1)

# #############################################################################
# Look at the results
lw = 2

svrs = [svr_rbf, svr_lin, svr_poly]
kernel_label = ['RBF', 'Linear', 'Polynomial']
model_color = ['m', 'c', 'g']

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 10), sharey=True)
for ix, svr in enumerate(svrs):
    axes[ix].plot(miv_X, svr.fit(miv_X_train, miv_y_train).predict(miv_X_test), color=model_color[ix], lw=lw,
                  label='{} model'.format(kernel_label[ix]))
    axes[ix].scatter(miv_X[svr.support_], y[svr.support_], facecolor="none",
                     edgecolor=model_color[ix], s=50,
                     label='{} support vectors'.format(kernel_label[ix]))
    axes[ix].scatter(miv_X[np.setdiff1d(np.arange(len(miv_X)), svr.support_)],
                     miv_y[np.setdiff1d(np.arange(len(miv_X)), svr.support_)],
                     facecolor="none", edgecolor="k", s=50,
                     label='other training data')
    axes[ix].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
                    ncol=1, fancybox=True, shadow=True)

fig.text(0.5, 0.04, 'data', ha='center', va='center')
fig.text(0.06, 0.5, 'target', ha='center', va='center', rotation='vertical')
fig.suptitle("Support Vector Regression", fontsize=14)
plt.show()

    
    
   
    # The coefficients
print('Coefficients: \n', regr.coefs_)
    # The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(miv_y_test, miv_y_pred))
    # Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(miv_y_test, miv_y_pred))
print('MAE score: %.2f' % mean_absolute_error(miv_y_test, miv_y_pred))
   
 
    
   
    
   
