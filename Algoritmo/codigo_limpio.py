import sys
import os
import csv
import subprocess
import uuid
from datetime import datetime as dt

try:
	name = uuid.uuid4()

	my_file_handle=open(str(name)+".txt","a")

	A  = os.popen('sudo iwlist wlp2s0 scanning | egrep "Cell |ESSID|Quality"').readlines()
	B = " ".join(str(x) for x in A) #Para pasar la lista con strings, a un solo string separado por espacios

	my_file_handle.write(B)
	my_file_handle = open(str(name)+ ".txt", "r")
	my_file_handle.close()

	with open(str(name)+ ".txt") as fp:
		lines = fp.readlines()

	ESSID = []
	MAC = []
	dBm=  []
	lim = len(lines)
	
	print('6: \n\tImpresión y almacenamiento de MAC de Dispositivos con su nombre')

	for i in range(1,lim,3):
			dBm.append(lines[i])

	for i in range(2,lim,3):
		ESSID.append(lines[i])

	for i in range(0,lim,3):
		MAC.append(lines[i])
	
	l = len(dBm)

	for i in range(0,l):
		dBm[i]= dBm[i].split()
		dBm[i]= dBm[i][2].replace("level=","")

		MAC[i] = MAC[i].split()
		MAC[i] = MAC[i][-1]
	
		ESSID[i]= ESSID[i].strip().replace("ESSID:","")
	
	with open('Potencia.csv', 'w',newline= '') as csvfile:
		filewriter = csv.writer(csvfile)
		for i in range(0,l):
			filewriter.writerow(['Potencia',])	
		
		filewriter.writerow(['MAC',MAC[0],MAC[1],MAC[2]])
		filewriter.writerow(['ESSID',ESSID[0],ESSID[1],ESSID[2]])
except IOError:
	print("File not found or path is incorrect")
finally:
	print("Exit")
