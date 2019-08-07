#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 13:12:18 2019

@author: aldo_mellado

Tengo que modificar el que las mediciones se copien en el mismo archivo, que una vez que se lee el nan se agregue un nivel de potencia bajo, -99, y se copie en el
Una vez hecho esto, necesito que 
"""

import os
import csv
import uuid
import time
import numpy as np
from tqdm import tqdm
from fixrows import fixrows
from getkey import getkey
from createcords import createcords


start_time = time.time()
flag=False
it=0
a = 1000


try:
	while(flag==False):
		print(f"i : {it}")
		with tqdm(total=a, desc="Writing on Potencia.csv", bar_format="{l_bar}{bar} [ iterations left: {remaining} ]") as pbar:
			try:
				for j in range(0,a):
					remaining = a - j

					name = uuid.uuid4()
					my_file1=open(str(name)+".txt",'a')

					A  = os.popen('sudo iwlist wlp2s0 scanning | egrep "Cell |ESSID|Quality"').readlines()
					B = " ".join(str(x) for x in A) #Para pasar la lista con strings, a un solo string separado por espacios (list comprehension)
					my_file1.write(B)

					with open(str(name)+ ".txt") as fp:
						lines = fp.readlines()

					ESSID = []
					MAC = []
					dBm=  []

					lim = len(lines)

					for i in range(1,lim,3):
						dBm.append(lines[i])

					for i in range(2,lim,3):
						ESSID.append(lines[i])

					for i in range(0,lim,3):
						MAC.append(lines[i])

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

				while(flag!=True):
					print("Press 'c' to continue or any key to quit")
					key = getkey()

					if key=='c':
						it+=1
						break
					elif key=='q':
						flag=True
						df = fixrows('Potencia') # A través de la función fixrows se ajustan las diferencias de elementos en las filas 
						num_row = np.shape(df)[0] # se toma la cantidad de filas que se obtuvieron luego de arreglarlo
						coords = createcords(num_row) # se crean un numero equivalente de pares ordenados para la cantidad de filas
						df= coords.join(df, how='right') # se unen ambas partes, los pares y las mediciones en un solo archivo
						df.to_csv('Datos.csv', index = None) #se exporta este a un archivo csv que será procesado por la red
						print(f"--- {time.time() - start_time} seconds ---\n")
					else:
						continue
			except:
				a-=1
except IOError:
	print("File not found or path is incorrect")
finally:
	print("Exit")
