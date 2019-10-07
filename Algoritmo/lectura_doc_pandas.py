#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 21:31:42 2019

@author: aldo_mellado
"""

import pandas as pd
import numpy as np

df = pd.read_excel('PLANILLA GESTION UEM 2018 OFICIAL.xlsx')
df1 = pd.DataFrame(df.iloc[:,1])
df2 = pd.DataFrame(df.iloc[:,20])
df2 = pd.DataFrame([str(j) for i,j in enumerate(df2['Fecha Inicio'])],dtype=object, columns = ['Fecha Inicio'])
df3 = df1.join(df2)


mes = input('\nIngrese Mes a buscar: ')
año = input('\nIngrese Año a buscar: ')

fecha  = año +"-" +mes

df_total_mes  = df3[df3['Fecha Inicio'].str.contains(fecha)]
num_trab = np.shape(df_total_mes)[0]

# Se buscan el índice de los elementos en la columna 'Estado UEM' que satisfacen la condicion de decir 'TRABAJO TERMINADO'
indexNames = df_total_mes[ df_total_mes['Estado UEM'] == 'TRABAJO TERMINADO'].index

# eliminar valores en índices que cumplieron la condicion en línea 28
df_total_mes.drop(indexNames , inplace=True)

# contar la cantidad de filas que tienen por estado 'Pendiente'
num_pendientes = np.shape(df_total_mes)[0]

#calculo porcentaje
porcentaje = round((num_trab - num_pendientes)/num_trab * 100,1)


import pygal

b_chart = pygal.SolidGauge(inner_radius=0.75)
b_chart.title = "Destiny Kill/Death Ratio"
b_chart.add("Trabajos Completados", porcentaje)
#b_chart.add("Trabajos Terminados", num_trab-num_pendientes)
#b_chart.add("Total Trabajos", num_trab)
b_chart.render_in_browser()

# =============================================================================
#                               Segundo Indicador (?)

# En esta parte haremos que se solicite un mes y una unidad en particular, 
# para entonces, mostrar las órdenes de trabajo asociadas a dicha unidad en dicho mes.
# =============================================================================
import pandas as pd
import numpy as np

df = pd.read_excel('PLANILLA GESTION UEM 2018 OFICIAL.xlsx')
df1 = pd.DataFrame(df.iloc[:,3]) #unidad

df1 = df1[df1['Servicio o Unidad'].notnull()] #filtra nan o NaN

df2 = pd.DataFrame(df.iloc[:,9]) #fecha recepción OT
df2 = pd.DataFrame([str(j) for i,j in enumerate(df2['Fecha recepcion OT'])],dtype=object, columns = ['Fecha recepcion OT']) #Fecha convertida a str para luego buscar fecha
df3 = df1.join(df2) #Unir ambos dataFrame para relacionarlos


mes = input('\nIngrese Mes a buscar: ')
año = input('\nIngrese Año a buscar: ')

fecha  = año +"-" +mes

unidades = np.unique(list(df1.iloc[1:,0])) #Me permite encontrar sin repeticion los nombres de los Servicios o Unidades disponibles

#opcion = input("Para filtrar los datos, ingrese:\na) Si desea filtrar por mes\nSi desea filtrar por año\nc)Si desea filtrar por Unidad\nd)Si desea filtrar por unidad y año\ne)Si desea filtrar por unidad y mes")

# Filtro por fecha
# Se ingresa el la fecha que se desea filtrar y aparecen las OT en esa fecha para
# todas las unidades

ot_mes  = df3[df3['Fecha recepcion OT' ].str.contains(fecha)]
num_trab = np.shape(ot_mes)[0]# Numero de OT generadas durante el mes y año indicados

# Crear un diccionario, que almacene la cantidad de ocurrencias u OT para esa unidad determinada
ocurrencias = {
        }

for i in unidades:
    num_ot_uni_mes = np.shape(ot_mes.loc[ot_mes['Servicio o Unidad'] == i])[0]
    print(f"num_ot_uni_mes = {num_ot_uni_mes}\t Servicio o Unidad = {i}")
    ocurrencias[i] = num_ot_uni_mes
    
num_to_filter = 3 #Cantidad de valores a filtrar
ocurr = pd.DataFrame(sorted(ocurrencias.items(), key = lambda x:x[1], reverse = True)).iloc[:num_to_filter,:]
flag =  any(ocurr.iloc[:,1])# verifica que la cantidad de OT para las unidades no sea 0

import pygal
b_chart = pygal.SolidGauge(inner_radius=0.75)
b_chart.title = "Num_SoU/Num tot Trab {} ".format(fecha)
if flag==True: #Si la cantidad de OT para las unidades no es 0, entonces:
    for i in range(num_to_filter):
        b_chart.add(ocurr.iloc[i,0], (ocurr.iloc[i,1]/num_trab)*100)
    b_chart.render_in_browser()
else: #en cambio, si no hubieron OT para ninguna unidad, entonces:
    print("\nEn esta fecha no existen OT")
#=============================================================================
#                           Tercer Indicador
# OT T1 Cerradas
#=============================================================================
nom_tec = ['CARLOS LOBOS','CRISTIAN REBOLLEDO',
           'GUIDO VICENCIO','IGNACIO VALDIVIA',
           'PABLO SILVA']

# (suma de todas las ordenes T1 generadas en el mes - sum ordenes T1 abiertas en el mes)/suma de todas las ordenes T1 generadas en el mes

import pandas as pd
import numpy as np

df = pd.read_excel('PLANILLA GESTION UEM 2018 OFICIAL.xlsx')
df1 = pd.DataFrame(df.iloc[:,14]) # Fecha Asignación

faas= np.unique(list(df1.iloc[1:,0]))
df1 = df1[df1['Fecha de asiganación'].notnull()] #filtra nan o NaN

df2 = pd.DataFrame(df.iloc[:,15]) #fecha recepción OT
df3 = pd.DataFrame(df.iloc[:,12])

df = df1.join(df2, how ='right')
df3 = df2.join(df2,how = 'left')

# Se buscan el índice de los elementos en la columna 'Estado UEM' que satisfacen la condicion de decir 'TRABAJO TERMINADO'
#for i in ['T1','T2','T3','T4']:
#    indexNames = [['Tipo de mantención'] == i].index

indexNames = [['Tipo de mantención'] == 'T1'].index

# eliminar valores en índices que cumplieron la condicion en línea 28
df.drop(indexNames , inplace=True)


mes = input('\nIngrese Mes a buscar: ')
año = input('\nIngrese Año a buscar: ')

fecha  = año +"-" +mes