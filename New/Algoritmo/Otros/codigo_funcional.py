#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 02:10:46 2019

@author: aldo_mellado
"""

import numpy as np
import pandas as pd

def header(msg):
    # 1. load hard-coded data into a dataframe
    print(msg)

header("1. load hard-coded data into a df")

df=  pd.DataFrame(
        [['Jan', 58, 42, 74, 22, 2.95],
         ['Feb', 61, 45, 78, 26, 3.02],
         ['Mar',65,48,84,25,2.34],
         ['Apr',67,50,92,28,1.02],
         ['May',71,53,98,35,0.48],
         ['Jun',75,56,107,41,0.11],
         ['Jul',77,58,105,44,0.0],
         ['Aug',77,59,102,43,0.03],
         ['Sep',77,57,103,40,0.17],
         ['Oct',73,54,96,34,0.81],
         ['Nov', 64,48,84,30,1.7],
         ['Dic', 58,42, 73,21,2.56]],
         index = [0,1,2,3,4,5,6,7,8,9,10,11],
         columns = ["Month",'avg_high','avg_low','record_high','record_low', 'avg_precipitation'])
print(df)

# 2. read text file into a dataframe
header("2. read text file into a dataframe")
#filename = 'Fremont_weather.txt'
#df= pd.read_csv(filename)
#print(df)

# 3. print first 5 or last 3 rows of df
header("3. df.head()")
print(df.head())

header("3 df.tail(3)")
df1 = df.tail(4)
print(df1.head())

# 4. get data types, index, columns, values
header("4. df.types")
print(df.dtypes)

header("4. df.index")
print(df.index)
print(np.(df.index))

header("4. df.columns")
print(df.columns)

header("4. df.values")
print(df.values)

header("5. df.describe")
print(df.describe())

#6. sort records by any column
header("6. df.sort_values('record_high', ascending = False)")
print(df.sort_values('record_high', ascending = False))

import pandas as pd
import numpy as np

df4 = pd.DataFrame({ 'id': [1,2,3], 'c1':[0,0,np.nan], 'c2': [np.nan,1,1]})
df = df[['id', 'c1', 'c2']]
df['num_nulls'] = df[['c1', 'c2']].isnull().sum(axis=1)
df.head()