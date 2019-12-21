#!/usr/bin/env python3 -*- coding: utf-8 -*-
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
#from fixrows import fixrows
#from createcords import createcords 

t = TicToc()


start_time = time.time()
flag=False
it = 0
a = 1
nombre = "Potencia_wlp2s0"

folder = os.getcwd()

if os.path.isdir(folder + '/Potencias') != True or os.path.isdir(folder + '/Archivos_Temporales') != True:
    os.mkdir('Potencias')
    os.mkdir('Archivos_Temporales')


t.tic()
name = uuid.uuid4()
my_file=open(folder + "/Archivos_Temporales/"+ str(name)+".txt",'a')

A  = os.popen('sudo iwlist wlp2s0 scan |egrep "Cell |ESSID|Quality"').readlines()

if len(A)==0:
    print("Interface down")
    time.sleep(10)
    print(f"time.sleep(10)")
    print("sudo ifconfig wlp2s0 up")
    os.popen('sudo ifconfig wlan0 up')
    pass

B = " ".join(str(x) for x in A) #Para pasar la lista con strings, a un solo string separado por espacios (list comprehension)

my_file.write(B)

my_file = open(folder+ "/Archivos_Temporales/" + str(name)+ ".txt", "r")

with open(folder+ "/Archivos_Temporales/" + str(name) + ".txt") as fp:
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

my_file.close()

l = len(dBm)

for i in range(0,l):
    dBm[i]= dBm[i].split()
    dBm[i]= int(dBm[i][2].replace("level=",""))
    
    MAC[i] = MAC[i].split()
    MAC[i] = MAC[i][-1]

    ESSID[i]= ESSID[i].strip().replace("ESSID:",'')

m_MAC_ESSID = np.array([ESSID[i]+'\n'+ MAC[i] for i in range(0,l)])

p_dBm = np.array(dBm)
print(p_dBm)