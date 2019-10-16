#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 02:20:43 2019

@author: aldo_mellado
"""
import pandas as pd
import numpy as np

def switch_demo(argument):
    '''Función que me permitirá filtrar los datos según sea la opción ingresada'''
    
    df = pd.read_excel('PLANILLA GESTION UEM 2018 OFICIAL.xlsx')
    df1 = pd.DataFrame(df.iloc[:,3]) #unidad

    df1 = df1[df1['Servicio o Unidad'].notnull()] #filtra nan o NaN

    df2 = pd.DataFrame(df.iloc[:,9]) #fecha recepción OT
    df2 = pd.DataFrame([str(j) for i,j in enumerate(df2['Fecha recepcion OT'])],dtype=object, columns = ['Fecha recepcion OT']) #Fecha convertida a str para luego buscar fecha
    df3 = df1.join(df2) #Unir ambos dataFrame para relacionarlos

    switcher = {
        'Mes': # Filtro por fecha
# Se ingresa el la fecha que se desea filtrar y aparecen las OT en esa fecha para
# todas las unidades

ot_mes  = df3[df3['Fecha recepcion OT' ].str.contains(fecha)]
num_trab = np.shape(ot_mes)[0],
        'Año': "March",
        'Unidad': "Feruary",
        'Unidad y Mes': "March",
        'Unidad y Año': "March",
    }
    print switcher.get(argument, "Invalid month")