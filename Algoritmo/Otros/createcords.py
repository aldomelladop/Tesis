#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 10:15:18 2019

@author: aldo_mellado
"""
import pandas as pd

def createcords(num_row):
# =============================================================================
#     Se tomaron las mediciones de cada baldosa para 9 filas con 50 mediciones por baldosa
# =============================================================================
    num_filas = 9
    num_mediciones = 50
    div = num_filas * num_mediciones
    aux = [[x,y] for x in range(num_row//div) for y in range(num_row//div)]
    
    aux1 = []
     
    for j,k in enumerate(aux):
        l=0
        while (l< num_mediciones):
            aux1.append(k)
            l+=1
            
    pos = pd.DataFrame(aux1) 
    pos.columns = ['X','Y']
    return(pos)


# =============================================================================
# b = [[x,y] for x in range(num_row) for y in range(num_row)]
# 
# 
# c = []
# n = 10
# for j,k in enumerate(b):
#     l=0
#     while (l< n):
#         c.append(k)
#         l+=1       
# =============================================================================
