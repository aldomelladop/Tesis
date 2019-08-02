#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 16:47:50 2019

@author: aldo_mellado
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 13:12:18 2019

@author: aldo_mellado

Tengo que modificar el que las mediciones se copien en el mismo archivo, que una vez que se lee el nan se agregue un nivel de potencia bajo, -99, y se copie en el
Una vez hecho esto, necesito que 
"""

import sys
import os
import csv
import uuid
import time
import subprocess
import numpy as np
import pandas as pd
from getkey import getkey, keys
from datetime import datetime as dt


start_time = time.time()
flag=False
it=0
a = 10

coord_x = []
coord_y = []

while(it<a):
    print(f"i : {it}")
    for j in range(0,a):
        print(f"\tj : {j+1}")
        coord_y.append(it)
        coord_x.append(j)
    it+=1

#df = pd.DataFrame({'X':coord_x,'Y':coord_y})
#coords= pd.DataFrame.from_dict(df)
    
coords= pd.DataFrame.from_dict({'X':coord_x,'Y':coord_y})

df= pd.read_csv('Potencias.csv',error_bad_lines=False)
df7 = coords.join(df, how='right')
df5 = df7.iloc[:,:df7.count(axis='columns').min()]


import time
import sys

toolbar_width = 40

# setup toolbar
sys.stdout.write("[%s]" % (" " * toolbar_width))
sys.stdout.flush()
sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['

for i in range(toolbar_width):
    time.sleep(0.1) # do real work here
    # update the bar
    sys.stdout.write("-")
    sys.stdout.flush()

sys.stdout.write("]\n") # this ends the progress bar