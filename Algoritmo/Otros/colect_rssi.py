#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 13:12:18 2019

@author: aldo_mellado

Tengo que modificar el que las mediciones se copien en el mismo archivo, que una vez que se lee el nan se agregue un nivel de potencia bajo, -99, y se copie en el
Una vez hecho esto, necesito que 
"""
import os
import csv
import uuid
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from getkey import getkey
from pytictoc import TicToc
from fixrows import fixrows
from createcords import createcords 
    
t = TicToc()


start_time = time.time()
flag=False
it = 0
a = 2000
nombre = "Potencia"

try:
    t.tic()
    while(flag==False):
        with tqdm(total=a, desc="Writing on  Potencia.csv", bar_format="{l_bar}{bar} [ remaining time: {remaining} ]") as pbar:
            t.tic()
            for j in range(0,a):
                
                name = uuid.uuid4()
                my_file=open(str(name)+".txt",'a')
                
                t.tic()
                A  = os.popen('sudo iwlist wlp2s0 scan |egrep "Cell |ESSID|Quality"').readlines()
                
                if len(A)==0:
                    print("Interface down")
                    time.sleep(10)
                    print(f"time.sleep(10)")
                    print("sudo ifconfig wlp2s0 up")
                    os.popen('sudo ifconfig wlp2s0 up')
                    continue
                else:
                    remaining = a-j
#                t.toc('iwlist = ')
                    

                t.tic()
                B = " ".join(str(x) for x in A) #Para pasar la lista con strings, a un solo string separado por espacios (list comprehension)
#                t.toc('join(str(x)) = ')

                t.tic()
                my_file.write(B)
#                t.toc('my_file.write(B) = ')

                my_file = open(str(name)+ ".txt", "r")

                t.tic()
                with open(str(name)+ ".txt") as fp:
                    lines = fp.readlines()
                
#                t.toc('t_readlines= ')

                ESSID = []
                MAC = []
                dBm=  []
                
                lim = len(lines)

                t.tic()
                for i in range(1,lim,3):
                    dBm.append(lines[i])
#                t.toc('t_dBm= ')

                t.tic()
                for i in range(2,lim,3):
                    ESSID.append(lines[i])

#                t.toc('t_ESSID= ')

                t.tic()
                for i in range(0,lim,3):
                    MAC.append(lines[i])

#                t.toc('t_MAC= ')

                my_file.close()
                
                l = len(dBm)

                t.tic()
                for i in range(0,l):
                    dBm[i]= dBm[i].split()
                    dBm[i]= dBm[i][2].replace("level=","")

                    MAC[i] = MAC[i].split()
                    MAC[i] = MAC[i][-1]

                    ESSID[i]= ESSID[i].strip().replace("ESSID:",'')

                m_MAC_ESSID = np.array([ESSID[i]+'\n'+ MAC[i] for i in range(0,l)])
                p_dBm = np.array(dBm)
#                t.toc('t_writing= ')

                t.tic()
                with open('Potencia.csv', 'a', newline = '') as csvfile:
                    filewriter = csv.writer(csvfile)

                    if it==0 and j==0:
                        filewriter.writerow(m_MAC_ESSID)
                    else:
                        filewriter.writerow(p_dBm)
#                t.toc('t_csv ')

                os.system("rm "+ str(name) +".txt")
                pbar.update(1)
                time.sleep(1)

#            t.toc('t_for = ')
        while(flag!=True):
            print("Press 'c' to continue or any key to quit")
            key = getkey()
            
            if key=='c':
                it+=1
                break
            elif key=='q':
                flag=True
                try:
                    t.tic()
                    df = fixrows(nombre) # A través de la función fixrows se ajustan las diferencias de elementos en las filas 
                    t.toc("fixing rows")
                    df.to_csv(nombre+ '_c.csv', index = None) #se exporta este a un archivo csv que será procesado por la red
                except Exception as e:  
                    print(e)
                    print("Error detectado")
                finally:
                    print(f"--- {((time.time() - start_time)/60):.2f} minutes ---\n")
            else:
                continue
except IOError:
    print("File not found or path is incorrect")
finally:
    print("Exit")