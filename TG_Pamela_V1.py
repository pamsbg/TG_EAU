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
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns

matriz_indices_vazoes = pd.read_csv('C:\\Users\\pamsb\\OneDrive\\Documentos\\TG Python\\Matriz_vazao_regress.csv', sep=';')


matriz_indices_vazoes.head()
miv = matriz_indices_vazoes
miv.loc[:,'vazao']

miv_y=miv.loc[:,'vazao']
# Use only one feature
miv_X = miv.loc[:,'SOLAR']

# Split the data into training/testing sets
miv_X_train = miv_X[:311].values.reshape(-1,1)
miv_X_test = miv_X[312:].values.reshape(-1,1)

# Split the targets into training/testing sets
miv_y_train = miv_y[:311].values.reshape(-1,1)
miv_y_test = miv_y[312:].values.reshape(-1,1)

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(miv_X_train, miv_y_train)

# Make predictions using the testing set
miv_y_pred = regr.predict(miv_X_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(miv_y_test, miv_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(miv_y_test, miv_y_pred))

# Plot outputs
plt.scatter(miv_X_test, miv_y_test,  color='black')
plt.plot(miv_X_test, miv_y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()


#MAPA DE CALOR

plt.figure(figsize=(10, 7))
sns.heatmap(miv.corr(),
            annot = True,
            fmt = '.2f',
            cmap='Blues')
plt.title('Correlação entre variáveis do dataset de vazão')
plt.show()




from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd


X = miv_X.values.reshape(-1,1)
y = miv_y
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=100)# Treinando modelo
model  = RandomForestClassifier()
model.fit(X_train, y_train)# Mostrando importância de cada feature
model.feature_importances_

importances = pd.Series(data=model.feature_importances_)
sns.barplot(x=importances, y=importances.index, orient='h').set_title('Importância de cada feature')