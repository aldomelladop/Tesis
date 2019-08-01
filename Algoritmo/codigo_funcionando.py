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
from getkey import getkey, keys
from datetime import datetime as dt


start_time = time.time()
flag=False
it=0
a = 100

try:
	while(flag==False):
		print(f"i : {it}")
		for j in range(0,a):
			print(f"\tj : {j+1}")

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
			coord_x = []
			coord_y = []

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
			df= pd.read_csv('Potencia.csv',error_bad_lines=False)
			df.to_csv('Potencia.csv', index = None)

		while(flag!=True):
			print("Press 'c' to continue or any key to quit")
			key = getkey()

			if key=='c':
				it+=1
				break
			elif key=='q':
				flag=True
				print(f"--- {time.time() - start_time} seconds ---\n")
			else:
				continue
except IOError:
	print("File not found or path is incorrect")
finally:
	print("Exit")