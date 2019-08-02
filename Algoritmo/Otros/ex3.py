import sys
import os
import csv
import subprocess

try:
	my_file_handle=open("test_file_2.txt","a")
	f = open("test.txt", 'w')
	sys.stdout = f
	B = [""]

	for i in range(0,2):
		A  = os.popen('sudo iwlist wlp2s0 scanning | egrep "Cell |ESSID|Quality"').readlines()
		B.append(" ".join(str(x) for x in A))
		my_file_handle.write('\n'+'['+str(i)+']'+'\n')
	
	
    
#	for i in range(0,len(B)-1):
#	print(B[i])

	my_file_handle.write(B)
	my_file_handle = open("test_file_2.txt", "r")
	my_file_handle.close()
	
#	f.close()

#	with open('person.csv', 'w') as csvFile:
#	writer = csv.writer(csvFile)
#	writer.writerows(A[0])
# 	csvFile.close()

except IOError:
	print("File not found or path is incorrect")
finally:
	print("Exit")
