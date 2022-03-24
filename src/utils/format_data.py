import re
import csv 
import pandas as pd 
import numpy as np
import xlsxwriter

def create_df(dataset, lines):
    bad_lines = 0
    data = [] # Lista para guardar la información de cada fila encontrada.
    n_cols = len(lines[0].split(';'))-1 # No consideramos el ID como parte de las columnas
    for i, line in enumerate(lines[1:]): # Recordar que la primera linea contiene el header.
        
        line = line.replace('\n','\t') # Sacamos los saltos de linea de los campos de texto para que no haya problema con el método reader de abajo.
        

        for val in csv.reader([line], delimiter=';'): # Separamos cada fila por el delimitador ;
            
            splitted_str = val
            
        if len(splitted_str)!=n_cols: # En caso que no calce la cantidad de columnas, saltamos esa fila.
            bad_lines+=1
            print(f"Skipping row {i}. The row consists of {len(splitted_str)} columns")
            continue
        data.append(splitted_str)
    
    print(len(data))
    print(len(data[0]))

    print(f'Number of bad samples in {dataset}: {bad_lines}')

    
    columns=['FECHA_INGRESO', 'ORIGEN_CASO', 'MERCADO_INGRESO', 'MERCADO_ANALISTA', 'TIPO_ENTIDAD', 'NOMBRE_ENTIDAD', 'TIPO_PRODUCTO', 'TIPO_MATERIA', 'DESCRIPCION_PROBLEMA', 'PETICION_SOLICITUD', 'CLASIFICACION_CIUDADANO']
    df = pd.DataFrame(data, columns = columns)
    print(df.head())
    # Según lo conversado en la reunión, si es que el valor de MERCADO ANALISTA está presente, entonces se considera por sobre el valor de MERCADO INGRESO.
    df.loc[df.MERCADO_ANALISTA=="Valores", "MERCADO_INGRESO"] = "VALORES"
    df.loc[df.MERCADO_ANALISTA=="Seguros", "MERCADO_INGRESO"] = "SEGUROS"
    df.loc[df.MERCADO_ANALISTA=="Bancos", "MERCADO_INGRESO"] = "BANCOS"
    

    df["Reclamo"] = df['DESCRIPCION_PROBLEMA'] + '. ' + df['PETICION_SOLICITUD']
    df = df[df['Reclamo'].str.strip()!='']
    #df = df[df['Reclamo'].apply(lambda x: len(x.split()) >= 5)]
    if dataset == 'bif':
        df = df[df.MERCADO_INGRESO != 'SEGUROS']
        df = df[df.MERCADO_INGRESO != 'VALORES']
        
    del df['FECHA_INGRESO']
    del df['ORIGEN_CASO']
    del df['MERCADO_ANALISTA']
    del df['DESCRIPCION_PROBLEMA']
    del df['PETICION_SOLICITUD']
    del df['CLASIFICACION_CIUDADANO']

    df = df.sample(frac = 1)
    df.drop_duplicates()
    df.to_excel(f"../../data/cleaned/{dataset}.xlsx", encoding='latin-1', engine='xlsxwriter') 
    return df

if __name__=='__main__':
    print(pd.__version__)
    print(xlsxwriter.__version__)
    # Lo primero que hacemos, es leer ambos archivos entregados para esta etapa. Se leen con codificación latin-1 ya que utf-8 no funciona.
    vys_file = open('../../data/raw/reclamos_VyS_hasta_2021.txt', 'r', encoding='latin-1').read() # Archivo con información de Valores y Seguros.
    bif_file = open('../../data/raw/reclamos_BIF_hasta_2021.txt', 'r', encoding='latin-1').read() # Archivo con información sobre Bancos.

    # Este paso es fundamental. Como el archivo .txt que venía en el supuesto formato .csv no se encontraba bien formateado, entonces no fue posible
    # realizar directamente la lectura con las librerías de pandas o csv, por lo tanto procedemos a obtener la información con un parser hecho por nosotros.
    # Lo que hacemos en la siguiente linea, es separar en el archivo cada vez que veamos un string de la forma "\nXXXXX;" ya que esto correspondería a un ejemplo
    # de ID_CASO, la primera columna en ambos archivos. 
    
    
    
    vys_lines = re.split(r"[\n]\d+[;]", vys_file) 
    bif_lines = re.split(r"[\n]\d+[;]", bif_file)

    vys_df = create_df('vys', vys_lines)
    bif_df = create_df('bif', bif_lines)
    
   

