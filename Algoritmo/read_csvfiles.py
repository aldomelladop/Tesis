#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 16:00:50 2019

@author: aldo_mellado
"""

import numpy as np
import pandas as pd
from fixrows import fixrows

# =============================================================================
df1 = fixrows('Potencia_r2_00')
df2 = fixrows('Potencia_r1_01')
df3 = fixrows('Potencia_r3_02')
# =============================================================================
#df1 = pd.read_csv('Potencia_r2_00_corregido.csv')
#df2 = pd.read_csv('Potencia_r1_01_corregido.csv')
#df3 = pd.read_csv('Potencia_r3_02_corregido.csv')

# =============================================================================
df4 = fixrows('Potencia_r2_10')
df5 = fixrows('Potencia_r1_11')
df6 = fixrows('Potencia_r3_12')
# =============================================================================
#df4 = pd.read_csv('Potencia_r2_10_corregido.csv')
#df5 = pd.read_csv('Potencia_r1_11_corregido.csv')
#df6 = pd.read_csv('Potencia_r3_12_corregido.csv')

# =============================================================================
df7 = fixrows('Potencia_r2_20')
df8 = fixrows('Potencia_r1_21')
df9 = fixrows('Potencia_r3_22')
# =============================================================================
#df7 = pd.read_csv('Potencia_r2_20_corregido.csv')
#df8 = pd.read_csv('Potencia_r1_21_corregido.csv')
#df9 = pd.read_csv('Potencia_r3_22_corregido.csv')