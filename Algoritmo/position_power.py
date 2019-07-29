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

# Seleccionar numero medio de potenbcias almacenadas

start_time = time.time()
flag=False
it=0

try:
    while(flag==False):
    	print(f"i : {it}")
    	for j in range(0,1001):
    		print(f"\t(x,y) : ({it},{j})")
    		#print(f"\tj : {j+1}")

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
	    	#print(f"\t\nl: {l}")
	    	
	    	for i in range(0,l):
	    		dBm[i]= dBm[i].split()
	    		dBm[i]= dBm[i][2].replace("level=","")

	    		MAC[i] = MAC[i].split()
	    		MAC[i] = MAC[i][-1]

	    		ESSID[i]= ESSID[i].strip().replace("ESSID:",'')

	    	#list comprehension with if/else: [f(x) if condition else g(x) for x in sequence]
	    	m_MAC_ESSID = np.array([ESSID[i]+'\n'+ MAC[i] for i in range(0,l)])

	    	m_MAC_ESSID = np.array(m_MAC_ESSID)
	    	#m_MAC_ESSID = np.array([ESSID[i]+'\n'+ MAC[i] for i in range(0,l)])

	    	x_y_pot.extend(dBm)
	    	x_y_pot.append(it)
	    	x_y_pot.append(j)
	    	x_y_pot = np.array(x_y_pot)
	    	
	    	
	    	with open('Potencia'+str(it)+'.csv', 'a', newline = '') as csvfile:
	    		filewriter = csv.writer(csvfile)

	    		if it==0 and j==0:
	    			filewriter.writerow(m_MAC_ESSID)
	    		else:
	    			filewriter.writerow(x_y_pot)
 
	    	os.system("rm "+ str(name) +".txt")

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
