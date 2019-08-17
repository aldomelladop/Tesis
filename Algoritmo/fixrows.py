import pandas as pd
import csv

# =========================================================================================================================
# Función creada para solucionar problema en la lectura de archivos csv 
#   con diferencia de cantidad de elementos por fila.
# =========================================================================================================================

def fixrows(name):

# =========================================================================================================================
# Lo que hace esta función es, leer el archivo, contar y almacenar en un vector la cantidad de elementos por fila.
# Posterior a esto, se escoge el número mínimo de elementos, pues, de este modo, se evita tener valores con 'NaN' en ellos.
# =========================================================================================================================
    min_width = 0
    amended_rows = []

    with open(name + '.csv', newline='', encoding='latin-1') as f:
        reader = csv.reader(f)
        width = []
        
        for row in reader:
            length = repr(row).count(",")
            width.append(length)
        
            if min(width)!=0:
                min_width = min(width)-1
                amended_rows = []

# =============================================================================
# Luego, se lee el archivo y redimensionan las filas para que calcen todas al
# número mínimo de elementos obtenido anteriormente.
# =============================================================================
                
    with open(name + '.csv', newline='', encoding='latin-1') as f:
        reader = csv.reader(f)
        
        for (i,row) in enumerate(reader):
            length2 = repr(row).count(",")
            
            if length2>min_width:
                row = row[:min_width]
#                print(f"ammended row lenght: {len(row)}")
                amended_rows.append(row)

# =============================================================================
# Para finalizar, se escriben estas líneas redimensionadas en un archivo aparte 
# para no perder datos que puedan ser utilizados a posteriori                    
# =============================================================================
                
    with open(name + '_corregido'+ '.csv','w', newline='', encoding='latin-1') as f:
        writer = csv.writer(f)
        writer.writerows(amended_rows)

    print(f"output file: {name}_corregido.csv")

# =================================================================
# Se retorna un tipo de dato DataFrame para simplicidad de trabajo
# =================================================================
    
    return(pd.read_csv(name + '_corregido'+ '.csv'))

fixrows.__doc__ = """ Lo que hace esta función es, leer el archivo, contar y almacenar en un vector la cantidad de elementos por fila.

                    Posterior a esto, se escoge el número mínimo de elementos, pues, de este modo, se evita tener valores con 'NaN' en ellos.
                    
                    fixrows(name):
                        
                        name = string, representa el nombre del archivo, sin extensión .csv, a procesar y corregir
                        
                    Se retorna un tipo de dato DataFrame para simplicidad de trabajo
                  """