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
pd.read_csv('Potencia.csv',error_bad_lines=False)
df4 = df.count(axis='columns')
df5 = df[:25]
df= pd.read_csv('Potencia0.csv',error_bad_lines=False)
df2 = df.dropna(how = 'any')
df3 = df.fillna(int(-99))
print(df.index)


X = df.iloc[:].values #Dependant variables
Y = df.iloc[:].values #Independant values


data = { 
    'A':{(1,1):['A1', 'A2', 'A3', 'A4', 'A5']},  
    'B':{(1,2):['B1', 'B2', 'B3', 'B4', 'B5']},  
    'C':{(1,3):['C1', 'C2', 'C3', 'C4', 'C5']},  
    'D':{(1,4):['D1', 'D2', 'D3', 'D4', 'D5']},  
    'E':{(1,5):['E1', 'E2', 'E3', 'E4', 'E5']} } 

aA =[{'A':'A1'}, {'A':'A2'}, {'A':'A3'}, {'A':'A4'}, {'A':'A5'}]

data1 = { 
    (1,1):['A1', 'A2', 'A3', 'A4', 'A5'],  
    (1,2):['B1', 'B2', 'B3', 'B4', 'B5'],
    (1,3):['C1', 'C2', 'C3', 'C4', 'C5'],  
    (1,4):['D1', 'D2', 'D3', 'D4', 'D5'],  
    (1,5):['E1', 'E2', 'E3', 'E4', 'E5'] } 

aA = pd.DataFrame.from_dict(aA)
data= pd.DataFrame.from_dict(data)
data1= pd.DataFrame.from_dict(data1)
# Convert the dictionary into DataFrame  
df = pd.DataFrame(data)
  
# Remove two columns name is 'C' and 'D' 
df.drop(['C', 'D'], axis = 1) 
  
# df.drop(columns =['C', 'D']) 


"