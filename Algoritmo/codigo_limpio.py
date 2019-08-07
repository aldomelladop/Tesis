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
from fixrows import fixrows
from createcords import createcords

start_time = time.time()
flag=False
it=0

try:
    while(flag==False):
    	print(f"i : {it}")
    	for j in range(0,10):
    		print(f"\t(x,y) : ({it},{j})")

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
	    	x_y_pot = []
	    	m_MAC_ESSID = []
	    	
	    	x_y_pot.append(it)
	    	x_y_pot.append(j)

	    	lim = len(lines)
	    	
	    	for i in range(1,lim,3):
	    		#print(f"dbm(lines[{i}]: {lines[i]}")
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

	    	m_MAC_ESSID.extend([ESSID[i]+'\n'+ MAC[i] for i in range(0,l)])
	    	m_MAC_ESSID = np.array(m_MAC_ESSID)
	    	#m_MAC_ESSID = np.array([ESSID[i]+'\n'+ MAC[i] for i in range(0,l)])

	    	p_dBm = np.array(dBm)
	    	
	    	with open('Potencia.csv', 'a', newline = '') as csvfile:
	    		filewriter = csv.writer(csvfile)

	    		if it==0 and j==0:
	    			filewriter.writerow(m_MAC_ESSID)
	    		else:
	    			filewriter.writerow(p_dBm)
 
	    	os.system("rm "+ str(name) +".txt")
            
    	while(flag!=True):
    		print("Press 'c' to continue or any key to quit")
    		key = getkey()
    		
    		if key=='c':
    			it+=1
    			break
    		elif key=='q':
    			flag=True
    			df = fixrows('Potencias') # A través de la función fixrows se ajustan las diferencias de elementos en las filas 
    			num_row = np.shape(df)[0] # se toma la cantidad de filas que se obtuvieron luego de arreglarlo
    			coords = createcords(num_row) # se crean un numero equivalente de pares ordenados para la cantidad de filas
    			df= coords.join(df, how='right') # se unen ambas partes, los pares y las mediciones en un solo archivo
    			df.to_csv('Datos.csv', index = None) #se exporta este a un archivo csv que será procesado por la red
    			print(f"--- {time.time() - start_time} seconds ---\n")
#                df = pd.read_csv('Datos.csv')
    		else:
    			continue
except IOError:
	print("File not found or path is incorrect")
finally:
	print("Exit")