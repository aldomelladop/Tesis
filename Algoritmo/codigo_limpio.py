import sys
import os
import csv
import uuid
import time
import numpy as np
import subprocess
from datetime import datetime as dt


start_time = time.time()

try:
	print(f"Start time: {start_time}")

	for i in range(0,3):
		print(f"i : {i}")
		name = uuid.uuid4()
		my_file_handle=open(str(name)+".txt","a")

		A  = os.popen('sudo iwlist wlp2s0 scanning | egrep "Cell |ESSID|Quality"').readlines()
		B = " ".join(str(x) for x in A) #Para pasar la lista con strings, a un solo string separado por espacios

		my_file_handle.write(B)
		my_file_handle = open(str(name)+ ".txt", "r")

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
		
		l = len(dBm)
		
		print(f"len(dBm) = {len(dBm)}")

		for i in range(0,l):
			dBm[i]= dBm[i].split()	
			dBm[i]= dBm[i][2].replace("level=","")

			MAC[i] = MAC[i].split()
			MAC[i] = MAC[i][-1]
		
			ESSID[i]= ESSID[i].strip().replace("ESSID:",'')
			#print(f"ESSID[i]: {ESSID[i]}")
		
		d_MAC = np.array(MAC)
		n_ESSID = np.array(ESSID)
		p_dBm = np.array(dBm)

		with open('Potencia.csv', 'w',newline = '') as csvfile:
			filewriter = csv.writer(csvfile)

			filewriter.writerow(d_MAC)
			filewriter.writerow(n_ESSID)
			filewriter.writerow(p_dBm)	

			my_file_handle.close()

	print(f"--- {time.time() - start_time} seconds ---")

except IOError:
	print("File not found or path is incorrect")
finally:
	print("Exit")
