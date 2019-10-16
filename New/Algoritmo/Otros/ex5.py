import sys
import os
import csv
import subprocess
import uuid
from datetime import datetime as dt


# ****************************************************** Notas **********************************************************

#	Recordar crear un nuevo archivo de texto logs, de lo contrario, cada vez se leeran mas y mas lineas  (listo)
#	¿Importa el orden de los datos?
#	¿Realmente necesito conocer el ESSID del dispositivo del cual recibo potencia, 
#		o solo me basta con caracterizar el espacio con base en la intesidad de señal recibida?

try:
	name = uuid.uuid4()

	print('\n0')
	print(f" \t{name}.txt")
	
	print('1')
	my_file_handle=open(str(name)+".txt","a")

	print('2')
#    A  = subprocess.popen('sudo iwlist wlp2s0 scanning | egrep "Cell |ESSID|Quality"').readlines()
	A  = os.popen('sudo iwlist wlp2s0 scanning | egrep "Cell |ESSID|Quality"').readlines()
	B = " ".join(str(x) for x in A) #Para pasar la lista con strings, a un solo string separado por espacios

	print('3')
	my_file_handle.write(B)
	my_file_handle = open(str(name)+ ".txt", "r")
	my_file_handle.close()

	print('4')

	with open(str(name)+ ".txt") as fp:
		lines = fp.readlines()

	print('5')

	#Vectores que almacenaran variables
	ESSID = []
	MAC = []
	dBm=  []
	lim = len(lines)
	
	print('6: \n\tImpresión y almacenamiento de MAC de Dispositivos con su nombre')

	for i in range(1,lim,3):
			dBm.append(lines[i])

	for i in range(2,lim,3):
	#	print(f"ESSID \t lines[{i}] = {lines[i]}")
		ESSID.append(lines[i])

	for i in range(0,lim,3):
	#	print(f"MAC\t lines[{i}] = {lines[i]}")
		MAC.append(lines[i])
	
	print('\n7\n')
	print(f"\nlen(lines): {len(lines)}\nlen(dBm): {len(dBm)}\nlen(MAC): {len(MAC)}\nlen(ESSID):{len(dBm)}\n")

	l = len(dBm)
	print(f"l: {l}\n")

	for i in range(0,l):
		dBm[i]= dBm[i].split()
		dBm[i]= dBm[i][2].replace("level=","")
		#print(f"dBm[{i}]: {dBm[i]}")

		MAC[i] = MAC[i].split()
		MAC[i] = MAC[i][-1]
		#print(f"MAC[0] = {MAC[i]}")
	
		ESSID[i]= ESSID[i].strip().replace("ESSID:","")
		#print(f"ESSID[{i}] = {ESSID[i]}")
		
	print('\n8\n')

	
	with open('Potencia.csv', 'w',newline= '') as csvfile:
		filewriter = csv.writer(csvfile)
		filewriter.writerow(['Potencia',dBm[2],dBm[1],dBm[2],dBm[3]])
		filewriter.writerow(['MAC',MAC[0],MAC[1],MAC[2]])
		filewriter.writerow(['ESSID',ESSID[0],ESSID[1],ESSID[2]])
except IOError:
	print("File not found or path is incorrect")
finally:
	print("Exit")
