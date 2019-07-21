import sys
import os
import csv
import subprocess
import uuid
import time
from datetime import datetime as dt

start_time = time.time()

try:
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
	
	#labels = ['MAC','ESSID','Potencia']
	#row = zip(MAC,ESSID,dBm)
	
	with open('Potencia.csv', 'w',newline= '') as csvfile:
		filewriter = csv.writer(csvfile)
		
		for rows in row:
			filewriter.writerow(rows)
		
		my_file_handle.close()

	print("--- %s seconds ---" % (time.time() - start_time))

except IOError:
	print("File not found or path is incorrect")
finally:
	print("Exit")


# while(flag!=True):
# 	    	print("Press 'c' to continue or any key to quit")
# 	    	key = getkey()

# 	    	if key=='c':
# 	    		it+=1
# 	    		break
# 	    	elif key=='q':
# 	    		flag=True
# 	    		print(f"--- {time.time() - start_time} seconds ---\n")
# 	    	else:
# 	    		continue