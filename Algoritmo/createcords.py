#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 10:15:18 2019

@author: aldo_mellado
"""
import pandas as pd

def createcords(num_row):
    pos = pd.DataFrame([[x,y] for x in range(num_row) for y in range(num_row)]) 
    pos.columns = ['X','Y']
    return(pos)
    
    