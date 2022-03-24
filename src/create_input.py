# En este archivo se crean los inputs para poder entrenar los clasificadores.
# Dadas las características del problema, en total se utilizarán 10 clasificadores.

import pandas as pd 



vys_df = pd.read_excel('../data/cleaned/vys.xlsx')
bif_df = pd.read_excel('../data/cleaned/bif.xlsx')

# 1. Clasificador (Bancos o Seguros/Valores)

#vys_df["MERCADO_INGRESO"] = 'SEGUROS-VALORES'
#vys_df = vys_df[["Reclamo", "MERCADO_INGRESO"]]




#bif_df = bif_df[["Reclamo", "MERCADO_INGRESO"]]



#dataset_1 = vys_df.append(bif_df, ignore_index=True)

#dataset_1 = dataset_1.sample(frac=1)
#print(dataset_1.head())
#print(dataset_1.shape)

#dataset_1.to_excel(f"../data/input_files/tipo_mercado_2_opciones.xlsx", encoding='latin-1', engine='xlsxwriter') 


# 2. Clasificador (Seguros o Valores)

#vys_df = pd.read_excel('../data/cleaned/vys.xlsx')
#vys_df = vys_df[["Reclamo", "MERCADO_INGRESO"]]


#dataset_2 = vys_df.sample(frac=1)

#dataset_2.to_excel(f"../data/input_files/tipo_mercado_seguros_valores.xlsx", encoding='latin-1', engine='xlsxwriter') 

#print(dataset_2.head())
#print(dataset_2.shape)

# 3. Clasificador Valores y Seguros - Tipo Entidad

#vys_df = pd.read_excel('../data/cleaned/vys.xlsx')
#vys_df = vys_df[["Reclamo", "TIPO_ENTIDAD"]]


#dataset_3 = vys_df.sample(frac=1)

#dataset_3.to_excel(f"../data/input_files/seguros_valores_tipo_entidad.xlsx", encoding='latin-1', engine='xlsxwriter') 

#print(dataset_3.head())
#print(dataset_3.shape)

# 4. Clasificador Valores y Seguros - Tipo Producto

#vys_df = pd.read_excel('../data/cleaned/vys.xlsx')
#vys_df = vys_df[["Reclamo", "TIPO_PRODUCTO"]]


#dataset_4 = vys_df.sample(frac=1)

#dataset_4.to_excel(f"../data/input_files/seguros_valores_tipo_producto.xlsx", encoding='latin-1', engine='xlsxwriter') 

#print(dataset_4.head())
#print(dataset_4.shape)

# 5. Clasificador Valores y Seguros - Tipo Materia

vys_df = pd.read_excel('../data/cleaned/vys.xlsx')
vys_df = vys_df[["Reclamo", "TIPO_MATERIA"]]

print(vys_df.head())

dataset_5 = vys_df.sample(frac=1)

dataset_5.to_excel(f"../data/input_files/seguros_valores_tipo_materia.xlsx", encoding='latin-1', engine='xlsxwriter') 

print(dataset_5.sample(1))
print(dataset_5.shape)

# 6. Clasificador Valores y Seguros - Nombre Entidad

#vys_df = pd.read_excel('../data/cleaned/vys.xlsx')
#vys_df = vys_df[["Reclamo", "NOMBRE_ENTIDAD"]]


#dataset_6 = vys_df.sample(frac=1)

#dataset_6.to_excel(f"../data/input_files/seguros_valores_nombre_entidad.xlsx", encoding='latin-1', engine='xlsxwriter') 

#print(dataset_6.head())
#print(dataset_6.shape)



# 7. Clasificador Bancos - Tipo Entidad

#bif_df = pd.read_excel('../data/cleaned/bif.xlsx')
#bif_df = bif_df[["Reclamo", "TIPO_ENTIDAD"]]


#dataset_7 = bif_df.sample(frac=1)

#dataset_7.to_excel(f"../data/input_files/bancos_tipo_entidad.xlsx", encoding='latin-1', engine='xlsxwriter') 

#print(dataset_3.head())
#print(dataset_3.shape)

# 8. Clasificador Bancos - Tipo Producto

#bif_df = pd.read_excel('../data/cleaned/bif.xlsx')
#bif_df = bif_df[["Reclamo", "TIPO_PRODUCTO"]]


#dataset_8 = bif_df.sample(frac=1)

#dataset_8.to_excel(f"../data/input_files/bancos_tipo_producto.xlsx", encoding='latin-1', engine='xlsxwriter') 



# 9. Clasificador Bancos - Tipo Materia

#bif_df = pd.read_excel('../data/cleaned/bif.xlsx')
#bif_df = bif_df[["Reclamo", "TIPO_MATERIA"]]


#dataset_9 = bif_df.sample(frac=1)

#dataset_9.to_excel(f"../data/input_files/bancos_tipo_materia.xlsx", encoding='latin-1', engine='xlsxwriter') 

# 10. Clasificador Bancos - Nombre Entidad

#bif_df = pd.read_excel('../data/cleaned/bif.xlsx')
#bif_df = bif_df[["Reclamo", "NOMBRE_ENTIDAD"]]


#dataset_10 = bif_df.sample(frac=1)

#dataset_10.to_excel(f"../data/input_files/bancos_nombre_entidad.xlsx", encoding='latin-1', engine='xlsxwriter') 