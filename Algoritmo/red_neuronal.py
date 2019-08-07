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
import matplotlib.pyplot as plt
from fixrows import fixrows

# =============================================================================
# Importing the dataset
# =============================================================================
df = fixrows('Potencias')
df = pd.read_csv('Potencia.csv')


X = df.iloc[:,2:].values #Independant values
Y = df.iloc[:,:1].values #Dependant variables
