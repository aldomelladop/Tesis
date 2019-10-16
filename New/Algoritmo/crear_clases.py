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
#from createcords import createcords
from merge_csv import fusionar_csv

# =============================================================================
# Importing the dataset
# =============================================================================
df = fixrows('Potencia_r2')
df1 = pd.read_csv('Potencia_r1_corregido.csv')
df = df.iloc[:4050,:]
num_row = np.shape(df)[0]

df3 = df + df2.iloc[1:,:]
coords  = [i for i in range(num_row//450) for j in range(num_row//9)]
coords = pd.DataFrame(coords,dtype=object, columns = ['pos'])
df = coords.join(df, how='right')

c0 = [i for i in range(0,150)]    + [i for i in range(450,600)]  + [i for i in range(900,1050)]
c1 = [i for i in range(150,300)]  + [i for i in range(600,750)]  + [i for i in range(1050,1200)]
c2 = [i for i in range(300,450)]  + [i for i in range(750,900)]  + [i for i in range(1200,1350)]

c3 = [i for i in range(1350,1500)]+ [i for i in range(1800,1950)]+ [i for i in range(2250,2400)]
c4 = [i for i in range(1500,1650)]+ [i for i in range(1950,2100)]+ [i for i in range(2400,2550)]
c5 = [i for i in range(1650,1800)]+ [i for i in range(2100,2250)]+ [i for i in range(2550,2700)]

c6 = [i for i in range(2700,2850)]+ [i for i in range(3150,3300)]+ [i for i in range(3600,3750)]
c7 = [i for i in range(2850,3000)]+ [i for i in range(3300,3450)]+ [i for i in range(3750,3900)]
c8 = [i for i in range(3000,3150)]+ [i for i in range(3450,3600)]+ [i for i in range(3900,4050)]

for i,j in df.iterrows():
    if i in c0:
        df.at[i, 'pos'] = 0
    if i in c1:
        df.at[i, 'pos'] = 1
    if i in c2:
        df.at[i, 'pos'] = 2
    if i in c3:
        df.at[i, 'pos'] = 3
    if i in c4:
        df.at[i, 'pos'] = 4
    if i in c5:
        df.at[i, 'pos'] = 5
    if i in c6:
        df.at[i, 'pos'] = 6
    if i in c7:
        df.at[i, 'pos'] = 7
    if i in c8:
        df.at[i, 'pos'] = 8

df.to_csv('corregido.csv', index = None)

df1 = fusionar_csv('Potencia_r1','Potencia_r2')