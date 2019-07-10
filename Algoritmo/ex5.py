import sys
import os
import csv
import subprocess
<<<<<<< HEAD
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
	ESSID = []
	MAC = []
	dBm=  []
	lim = len(lines)-1
	
	print(f"lim: {lim}")

	print('6: \tImpresión y almacenamiento de MAC de Dispositivos con su nombre')

	for i in range(0,lim):
		print(f"\t lines[{i}] = {lines[i]}")

	for i in range(1,lim,3):
		print(f"dBm \t lines[{i}] = {lines[i]}")
		dBm.append(lines[i])

	for i in range(2,lim,3):
		print(f"ESSID \t lines[{i}] = {lines[i]}")
		ESSID.append(lines[i])

	for i in range(0,lim,3):
		print(f"MAC\t lines[{i}] = {lines[i]}")
		MAC.append(lines[i])
	
	print(f"\nlen(lines): {len(lines)}\nlen(dBm): {len(dBm)}\nlen(MAC): {len(MAC)}\nlen(ESSID):{len(dBm)}\n")

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
=======

try:
    print('1')
    my_file_handle=open("test_file_3.txt","a")
 
    print('2')
#    A  = subprocess.popen('sudo iwlist wlp2s0 scanning | egrep "Cell |ESSID|Quality"').readlines()
    A  = os.popen('sudo iwlist wlp2s0 scanning | egrep "Cell |ESSID|Quality"').readlines()
    B = " ".join(str(x) for x in A) #Para pasar la lista con strings, a un solo string separado por espacios

    print('3')
    my_file_handle.write(B)
    my_file_handle = open("test_file_3.txt", "r")
    my_file_handle.close()

    print('4')

    with open('test_file_3.txt') as fp:
    	lines = fp.readlines()

    print('5')

    aux=[]
    aux1= []
    Crudo = []
    Potencias = []

    lim = len(lines)-1
   	
    print('6')
    for i in range(1,lim,3):
    	print(f"lines[{i}] = {lines[i]}")
    	Crudo.append((lines[i]).strip()) #Elimina los espacios que existían al comienzo del string
    
    print('7')
    
    for x,y in enumerate(Crudo):
    	print(f"[{x}]: [{y}]")
    	aux[x]= y.split()
    
    print('\n9')
    for x,y in enumerate(aux):
    	print(f"[{x}]: {y}")
    	
    print(aux[0])
    #print(f"Potencias[0] = {aux[0][2]}")	
    #print(f"Potencias[0] = {Potencias[0][2]}")
    #print(f"Potencias[1] = {Potencias[1][2]}")	




    #print(f"Par[0]: {Par[0]}\nPar[1]: {Par[1]}\nPar[2]: {Par[2]}")
    #print(f"Impar[0]: {Impar[0]}\nImpar[1]: {Impar[1]}\nImpar[2]: {Impar[2]}")

	#with open('person.csv', 'w') as csvFile:
	#	writer = csv.writer(csvFile)
    #	writer.writerows(A[0])
    # 	csvFile.close()
    #lineList = [line.rstrip('\n') for line in open("test_file_3.txt")]
	"""
>>>>>>> f41284856b2e0c07afc11c1e09963e4da817b529

except IOError:
	print("File not found or path is incorrect")
finally:
	print("Exit")
