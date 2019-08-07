import pandas as pd
import csv

nombre = 'Datos'

with open(nombre + '.csv', newline='', encoding='latin-1') as f:
    reader = csv.reader(f)
    max_width = 0    
    width = []
    
    for row in reader:
        length = repr(row).count(",")
        width.append(length)
    
        if min(width)!=0:
            min_width = min(width)-1
            amended_rows = []
            reader = csv.reader(f)
            
            for row in reader:
                length2 = repr(row).count(",")
                if length2>min_width :
                    row = row[:min_width]
                    print(f"ammended row lenght: {len(row)}")
                    amended_rows.append(row)
        
        
with open(nombre + '_corregido'+ '.csv','w', newline='', encoding='latin-1') as f:    
    writer = csv.writer(f)
    writer.writerows(amended_rows)