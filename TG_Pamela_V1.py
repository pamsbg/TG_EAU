# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 20:37:37 2019

@author: pamsb
"""

 # import numpy and pandas, and DataFrame / Series
import numpy as np
import pandas as pd
from pandas import DataFrame, Series

matriz_indices = pd.read_csv('C:\\Users\\pamsb\\OneDrive\\Documentos\\TG Python\\matriz_indicestele1979a2014_soMEI.txt')
vazoes = pd.read_csv('C:\\Users\\pamsb\\OneDrive\\Documentos\\TG Python\\vazao_clean.csv')

vazoes.head()
vazao_index = vazoes.set_index("Ano",drop=False)
vazao_treino = vazoes.iloc[[<2004],:]


