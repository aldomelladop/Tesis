import sys
import os
import csv
import subprocess

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

except IOError:
	print("File not found or path is incorrect")
finally:
	print("Exit")
