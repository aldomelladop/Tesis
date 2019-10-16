#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 13:12:18 2019

@author: aldo_mellado

Tengo que modificar el que las mediciones se copien en el mismo archivo, que una vez que se lee el nan se agregue un nivel de potencia bajo, -99, y se copie en el
Una vez hecho esto, necesito que 
"""

import sys
import os
import csv
import uuid
import time
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm
from getkey import getkey, keys
from datetime import datetime as dt

start_time = time.time()
flag=False
it=0
a = 50
coord_x = []
coord_y = []

try:
	while(flag==False):
		print(f"i : {it}")
		print(f"Writing on Potencias.csv \t")
		for i in tqdm(a):
			for j in range(0,a):
				remaining = a - j
			
				name = uuid.uuid4()
				my_file=open("respaldo.txt",'a')
				my_file1=open(str(name)+".txt",'a')
				
				A  = os.popen('sudo iwlist wlp2s0 scanning | egrep "Cell |ESSID|Quality"').readlines()
				B = " ".join(str(x) for x in A) #Para pasar la lista con strings, a un solo string separado por espacios (list comprehension)

				my_file.write(B)
				my_file1.write(B)

				my_file=open("respaldo.txt",'r')
				my_file1 = open(str(name)+ ".txt", "r")

				with open(str(name)+ ".txt") as fp:
					lines = fp.readlines()

				ESSID = []
				MAC = []
				dBm=  []
				coord_x.append(it)
				coord_y.append(j)

				lim = len(lines)

				for i in range(1,lim,3):
					dBm.append(lines[i])
				
				for i in range(2,lim,3):
					ESSID.append(lines[i])

				for i in range(0,lim,3):
					MAC.append(lines[i])

				my_file.close()
				my_file1.close()
				l = len(dBm)

				for i in range(0,l):
					dBm[i]= dBm[i].split()
					dBm[i]= dBm[i][2].replace("level=","")

					MAC[i] = MAC[i].split()
					MAC[i] = MAC[i][-1]

					ESSID[i]= ESSID[i].strip().replace("ESSID:",'')

				m_MAC_ESSID = np.array([ESSID[i]+'\n'+ MAC[i] for i in range(0,l)])
				p_dBm = np.array(dBm)

				with open('Potencia.csv', 'a', newline = '') as csvfile:
					filewriter = csv.writer(csvfile)

					if it==0 and j==0:
						filewriter.writerow(m_MAC_ESSID)
					else:
						filewriter.writerow(p_dBm)

				os.system("rm "+ str(name) +".txt")
				pbar.update(1)

				"""
				#en esta parte, se pretende transformar las listas convertidas a arrays en DataFrames, para luego poder hacer un recuento del numero 
				#mínimo de elementos no nulos en filas y con dicho número, cortar la cantidad de datos admitidos. Finalmente, se hace un append de 
				#las coordenadas de posición en que se tomaron dichas muestras.

				df= pd.read_csv('Potencias.csv',error_bad_lines=False)
				num = df.count(axis='columns')
				minimo= df4.min()
				df = df.iloc[:,:minimo]
				coord = pd.DataFrame()

				df2 = pd.DataFrame({"x":coord_x,"y":coord_y}) 
				df = coord.append(df2, ignore_index = True)
				"""
				#time.sleep(3)
				
			
		while(flag!=True):
			print("Press 'c' to continue or any key to quit")
			key = getkey()

			if key=='c':
				it+=1
				break
			elif key=='q':
				flag=True
				coords= pd.DataFrame.from_dict({'X':coord_x,'Y':coord_y})
				print(coords)
				df= pd.read_csv('Potencias.csv', error_bad_lines=False)
				df7 = coords.join(df, how='right')
				df5 = df7.iloc[:,:df7.count(axis='columns').min()]
				df5.to_csv('Datos.csv', index = None)
                
				print(f"--- {time.time() - start_time} seconds ---\n")
			else:
				continue
except IOError:
	print("File not found or path is incorrect")
finally:
	print("Exit")
