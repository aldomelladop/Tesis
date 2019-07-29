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
import matplotlib.pyplot as plt
import pandas as pd

# =============================================================================
# Importing the dataset
# =============================================================================
#df= pd.read_csv('Potencia0.csv',error_bad_lines=False)
df= pd.read_csv('Potencia.csv',error_bad_lines=False)
df= pd.read_csv('Potencia0.csv',error_bad_lines=False)
df2 = df.dropna(how = 'any')
df3 = df.fillna(int(-99))
print(df.index)


X = df.iloc[:].values #Dependant variables
Y = df.iloc[:].values #Independant values




"