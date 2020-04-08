# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 18:02:41 2020

@author: pamsb
"""

import pandas as pd
import numpy as np

base = pd.read_csv('credit-data.csv')
base.loc[base.age<0, 'age'] = 40.92
previsores = base.iloc[:,0:4].values
classe= base.iloc[:,4].values


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer = imputer.fit(previsores[:, 1:4])
previsores[:, 1:4] = imputer.transform(previsores[:, 1:4])
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)
from sklearn.naive_bayes import GaussianNB
previsores.shape
previsores.shape[0]
b = np.zeros(shape=(previsores.shape[0],1))

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
kfold = StratifiedKFold(n_splits = 10, shuffle = True, random_state =5)
resultados = []
matrizes = []
for indice_treinamento, indice_teste in kfold.split(previsores, np.zeros(shape=( previsores.shape[0],1))):
 #print('Indice treinamento: ', indice_treinamento, 'Indice teste: ', indice_teste)
 classificador = GaussianNB()
 classificador.fit(previsores[indice_treinamento], classe[indice_treinamento])
 previsoes = classificador.predict(previsores[indice_teste])
 precisao = accuracy_score(classe[indice_teste], previsoes)
 matrizes.append(confusion_matrix(classe[indice_teste], previsoes))
 resultados.append(precisao)

matrizes_final = np.mean(matrizes,axis=0)
resultados = np.array(resultados)
resultados.mean()

