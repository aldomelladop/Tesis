#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 10:24:48 2019

@author: aldo_mellado
"""
# =============================================================================
# Importing the libraries
# =============================================================================
import numpy as np
import pandas as pd
from fixrows import fixrows
from createcords import createcords

# =============================================================================
# Importing the dataset
# =============================================================================

df = fixrows('Potencia')
num_row = np.shape(df)[0]

coords = createcords(num_row)
# =============================================================================
#  Creating the Second Dataframe using dictionary 
# =============================================================================
    # num_row = np.shape(df)[0]-1  representa la cantidad de muestras tomadas para una posición dada

#pos = pd.DataFrame({"X":[i for i in range(num_row)],"Y":[i for i in range(0,num_row)]}) 
df = coords.join(df, how='right')
df.to_csv('Datos.csv', index = None)
df = pd.read_csv('Potencia_corregido.csv')

#df4 = df7.count(axis='columns')
#df5 = df7.iloc[:,:df4.min()]

X = df.iloc[:,2:].values #Independant values
Y = df.iloc[:,:1].values #Dependant variables

a =[[x,y] for x in range(num_row) for y in range(num_row)]

num= input()
a = [[x,y] for x in range(int(num)) for y in range(int(num))]
a = pd.DataFrame(a)

from createcords import createcords

b = createcords(num_row)


