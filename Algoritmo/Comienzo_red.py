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
df= pd.read_csv('Potencias.csv',error_bad_lines=False)

X = df.iloc[:].values #Dependant variables
Y = df.iloc[:].values #Independant values

"""
data = {'A':{(1,1):['A1', 'A2', 'A3', 'A4', 'A5']},'B':{(1,2):['B1', 'B2', 'B3', 'B4', 'B5']},'C':{(1,3):['C1', 'C2', 'C3', 'C4', 'C5']},  
    'D':{(1,4):['D1', 'D2', 'D3', 'D4', 'D5']},'E':{(1,5):['E1', 'E2', 'E3', 'E4', 'E5']} } 

a = {'A':{(1,1):'A1'}}
aA = {'A':{(1,1):'A1'},'A':{(1,2):'A2'},'A':{(1,3):'A3'},'A':{(1,4):'A4'},'A':{(1,5):'A5'},
      'B':{(1,1):'B1'},'B':{(1,2):'B2'},'B':{(1,3):'B3'},'B':{(1,4):'B4'},'B':{(1,5):'B5'},
      'C':{(1,1):'C1'},'C':{(1,2):'C2'},'C':{(1,3):'C3'},'C':{(1,4):'C4'},'C':{(1,5):'C5'},
      'D':{(1,1):'D1'},'D':{(1,2):'D2'},'D':{(1,3):'D3'},'D':{(1,4):'E5'},'D':{(1,5):'D5'},
      'E':{(1,1):'E1'},'E':{(1,2):'E2'},'E':{(1,3):'E3'},'E':{(1,4):'E5'},'E':{(1,5):'E5'},
      }

data1 = { (1,1):['A1', 'A2', 'A3', 'A4', 'A5'], (1,2):['B1', 'B2', 'B3', 'B4', 'B5'],(1,3):['C1', 'C2', 'C3', 'C4', 'C5'],
        (1,4):['D1', 'D2', 'D3', 'D4', 'D5'], (1,5):['E1', 'E2', 'E3', 'E4', 'E5']} 
A = pd.DataFrame.from_dict(aA)
data= pd.DataFrame.from_dict(data)
data1= pd.DataFrame.from_dict(data1)
"""
# Convert the dictionary into DataFrame  
df = pd.DataFrame(data)
 
# Creating the first Dataframe using dictionary 
df1 = pd.DataFrame({"a":[1, 2, 3, 4], "b":[5, 6, 7, 8]})
  
# Creating the Second Dataframe using dictionary 
df2 = pd.DataFrame({"X":[1, 1, 1],"Y":[0, 1, 2]}) 
  
# for appending df2 at the end of df1 
#df6 = df5.append(df2, ignore_index = True, sort=True)
df7 = df2.join(df, how='right')
df4 = df7.count(axis='columns')
df5 = df7.iloc[:,:df4.min()]


 


