import sys
import os
import csv
import subprocess
from datetime import date

# ****************************************************** Notas **********************************************************

#	Recordar crear un nuevo archivo de texto logs, de lo contrario, cada vez se leeran mas y mas lineas  (listo)
#	¿Importa el orden de los datos?
#	¿Realmente necesito conocer el ESSID del dispositivo del cual recibo potencia, 
#		o solo me basta con caracterizar el espacio con base en la intesidad de señal recibida?

try:
	today = date.today()
	print('1')
	my_file_handle=open(str(today)+".txt","a")

	print('2')
#    A  = subprocess.popen('sudo iwlist wlp2s0 scanning | egrep "Cell |ESSID|Quality"').readlines()
	A  = os.popen('sudo iwlist wlp2s0 scanning | egrep "Cell |ESSID|Quality"').readlines()
	B = " ".join(str(x) for x in A) #Para pasar la lista con strings, a un solo string separado por espacios

	print('3')
	my_file_handle.write(B)
	my_file_handle = open(str(today)+".txt", "r")
	my_file_handle.close()

	print('4')

	with open(str(today)+".txt") as fp:
		lines = fp.readlines()

	print('5')

	#Vectores que almacenaran variables
	aux = []
	ESSID = []
	MAC = []
	dBm=  []
	lim = len(lines)-1
	
	print(f"\nlim: {lim}\n")

	print('6: \nImpresión y almacenamiento de MAC de Dispositivos con su nombre\n')

	#for i in range(0,lim):
		#print(f"\t lines[{i}] = {lines[i]}")

	# for i in range(1,lim,3):
	# 	print(f"dBm \t lines[{i}] = {lines[i]}")
	# 	aux[i]= lines[i].split()
	# 	dBm.append(aux[i][2])
	
	# aux.clear()

	print('\n7\n')
	for i in range(2,lim,3):
		print(f"ESSID \t lines[{i}] = {lines[i]}")
		aux[i]= lines[i].strip().replace("ESSID:","")
		ESSID.append(aux[i])

	aux.clear()

	print('\n8\n')
	for i in range(0,lim,3):
		print(f"MAC\t lines[{i}] = {lines[i]}")
		aux[i] = lines[i].split()
		MAC.append(lines[i][-1])
	
	print(f"\nlen(lines): {len(lines)}\nlen(dBm): {len(dBm)}\nlen(MAC): {len(MAC)}\nlen(ESSID):{len(dBm)}\n")

	print('\n9\n')
	"""
	l = len(dBm)-1

	for i in range(0,l):
		dBm[i]= dBm[i].split()
		#print(f"dBm[0][2:]: {dBm[0][2]}")

		ESSID[i]= ESSID[i].strip().replace("ESSID:","")
		#print(f"ESSID[0] = {ESSID[0]}")
	
		MAC[i] = MAC[i].split()
		#print(f"MAC[0] = {MAC[0][-1]}")
		if i>30:
			print(f"\ndBm[{i}][2:]\t level: {dBm[i][2]}")
			print(f"ESSID[{i}][2:]: {ESSID[i]}")
			print(f"MAC[{i}][2:]: {MAC[i][-1]}\n")



#	dBm[0]= dBm[0].split()
#	print(f"dBm[0][2:]: {dBm[0][2]}")
#
#	ESSID[0]= ESSID[0].strip().replace("ESSID:","")
#	print(f"ESSID[0] = {ESSID[0]}")
#	
#	MAC[0] = MAC[0].split()
#	print(f"MAC[0] = {MAC[0][-1]}")

	
	#csvData = [['ESSID', 'MAC','level [dBm]'], ['Peter', '22'], ['Jasmine', '21'], ['Sam', '24']]
	#with open('Potencias.csv', 'w') as csvFile:
    #riter = csv.writer(csvFile)
    #riter.writerows(csvData)
	#csvFile.close()
	"""
except IOError:
	print("File not found or path is incorrect")
finally:
	print("Exit")
