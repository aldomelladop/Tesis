#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 03:05:31 2019

@author: aldo_mellado
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 15:41:02 2019

@author: aldo_mellado
"""

import csv
import shutil, os
import pandas as pd
from fixrows import fixrows

def copiar_archivo(file):
    ruta = os.getcwd() + os.sep
    origen = ruta + str(file)+ '.csv'
    destino= ruta + str(file)+ '_copia.csv'
    
    if os.path.exists(origen):
        try:
            archivo = shutil.copy(origen, destino)
            print('Copiado...', archivo)
        except:
            print('Error en la copia')

# =============================================================================
# def fusionar_csv (file1,file2):
#     '''añadir filas de un archivo csv a otro, evitando que estas se superpongan 
#     con otros datos '''
#     
#     for file in [file1,file2]:
#         copiar_archivo(file)
#         print("Archivo copiado")
#         df = fixrows(str(file)+'_copia')
#         print("Archivo redimensionado")
#         
#         df = pd.read_csv(str(file) + '_copia_corregido.csv')        
#         
#         if file==file2:
#             df.columns = df.iloc[1]        
#             
#         df.to_csv(str(file) + '_copia_corregido.csv', index = None)
#         print(f"file : {str(file)}")
#             
#     with open('potencias_fusionado.csv' ,'w') as outFile:
#         fileWriter = csv.writer(outFile)
#         for file in [file1,file2]:
#             with open(str(file) + '_copia'+ '_corregido.csv','r') as inFile:
#                 fileReader = csv.reader(inFile)
#                 for row in fileReader:
#                     fileWriter.writerow(row)
#     
#     print(f"Archivos fusionados: {str(file1) + '_copia.csv'}.csv")     
# =============================================================================
def fusionar_csv (*files):
    '''añadir filas de un archivo csv a otro, evitando que estas se superpongan 
    con otros datos '''
    
    j = 0
    
    for file in files:
        copiar_archivo(file)
        print("Archivo copiado")
        df = fixrows(str(file)+'_copia')
        print("Archivo redimensionado")
        
        df = pd.read_csv(str(file) + '_copia_corregido.csv')        
        
        if j!=0:
            df.columns = df.iloc[1]

        df.to_csv(str(file) + '_copia_corregido.csv', index = None)
        print(f"file : {str(file)}")
        j+=1
            
    with open('potencias_fusionado.csv' ,'w') as outFile:
        fileWriter = csv.writer(outFile)
        for file in files:
            with open(str(file) + '_copia'+ '_corregido.csv','r') as inFile:
                fileReader = csv.reader(inFile)
                for row in fileReader:
                    fileWriter.writerow(row)
    
    print(f"Archivos fusionados: {str(files)}")

# =============================================================================
# with open('user.csv', '') as userFilem:
#     userFileReader = csv.reader(userFile)
#     for row in userFileReader:
#         userList.append(row)
# =============================================================================