# En este archivo se crean los inputs para poder entrenar los clasificadores.
# Dadas las características del problema, en total se utilizarán 10 clasificadores.

import pandas as pd 



vys_df = pd.read_excel('../../data/cleaned/vys.xlsx')
bif_df = pd.read_excel('../../data/cleaned/bif.xlsx')

# 1. Clasificador (Bancos o Seguros/Valores)

vys_df["MERCADO_INGRESO"] = 'SEGUROS-VALORES'
vys_df = vys_df[["Reclamo", "MERCADO_INGRESO"]]
bif_df = bif_df[["Reclamo", "MERCADO_INGRESO"]]
df = vys_df.append(bif_df, ignore_index=True)
df = df[df['MERCADO_INGRESO'].notna()] 
df.to_excel("../../data/input_files/tipo_mercado.xlsx", encoding='latin-1', engine='xlsxwriter') 

print(df.head())

# 2. Clasificador (Seguros o Valores)

vys_df = pd.read_excel('../../data/cleaned/vys.xlsx')
vys_df = vys_df[["Reclamo", "MERCADO_INGRESO"]]
vys_df = vys_df[vys_df['MERCADO_INGRESO'].notna()] 
vys_df.to_excel("../../data/input_files/tipo_mercado_seguros_valores.xlsx", encoding='latin-1', engine='xlsxwriter') 

print(vys_df.head())

# 3. Clasificador Valores y Seguros - Tipo Entidad

vys_df = pd.read_excel('../../data/cleaned/vys.xlsx')
vys_df = vys_df[["Reclamo", "TIPO_ENTIDAD"]]
vys_df = vys_df[vys_df['TIPO_ENTIDAD'].notna()] 
vys_df.to_excel("../../data/input_files/seguros_valores_tipo_entidad.xlsx", encoding='latin-1', engine='xlsxwriter') 

print(vys_df.head())

# 4. Clasificador Valores y Seguros - Tipo Producto

vys_df = pd.read_excel('../../data/cleaned/vys.xlsx')
vys_df = vys_df[["Reclamo", "TIPO_PRODUCTO"]]
vys_df = vys_df[vys_df['TIPO_PRODUCTO'].notna()] 
vys_df.to_excel("../../data/input_files/seguros_valores_tipo_producto.xlsx", encoding='latin-1', engine='xlsxwriter') 

print(vys_df.head())
# 5. Clasificador Valores y Seguros - Tipo Materia

vys_df = pd.read_excel('../../data/cleaned/vys.xlsx')
vys_df = vys_df[["Reclamo", "TIPO_MATERIA"]]
vys_df = vys_df[vys_df['TIPO_MATERIA'].notna()] 
vys_df.to_excel("../../data/input_files/seguros_valores_tipo_materia.xlsx", encoding='latin-1', engine='xlsxwriter') 

print(vys_df.head())


# 6. Clasificador Valores y Seguros - Nombre Entidad

vys_df = pd.read_excel('../../data/cleaned/vys.xlsx')
vys_df = vys_df[["Reclamo", "NOMBRE_ENTIDAD"]]
vys_df = vys_df[vys_df['NOMBRE_ENTIDAD'].notna()] 
vys_df.to_excel("../../data/input_files/seguros_valores_nombre_entidad.xlsx", encoding='latin-1', engine='xlsxwriter') 

print(vys_df.head())


# 7. Clasificador Bancos - Tipo Entidad

bif_df = pd.read_excel('../../data/cleaned/bif.xlsx')
bif_df = bif_df[["Reclamo", "TIPO_ENTIDAD"]]
bif_df = bif_df[bif_df['TIPO_ENTIDAD'].notna()] 
bif_df.to_excel("../../data/input_files/bancos_tipo_entidad.xlsx", encoding='latin-1', engine='xlsxwriter') 

print(bif_df.head())

# 8. Clasificador Bancos - Tipo Producto

bif_df = pd.read_excel('../../data/cleaned/bif.xlsx')
bif_df = bif_df[["Reclamo", "TIPO_PRODUCTO"]]
bif_df = bif_df[bif_df['TIPO_PRODUCTO'].notna()]
bif_df.to_excel("../../data/input_files/bancos_tipo_producto.xlsx", encoding='latin-1', engine='xlsxwriter') 
 
print(bif_df.head())

# 9. Clasificador Bancos - Tipo Materia

bif_df = pd.read_excel('../../data/cleaned/bif.xlsx')
bif_df = bif_df[["Reclamo", "TIPO_MATERIA"]]
bif_df = bif_df[bif_df['TIPO_MATERIA'].notna()]
bif_df.to_excel("../../data/input_files/bancos_tipo_materia.xlsx", encoding='latin-1', engine='xlsxwriter')
 
print(bif_df.head())

# 10. Clasificador Bancos - Nombre Entidad

bif_df = pd.read_excel('../../data/cleaned/bif.xlsx')
bif_df = bif_df[["Reclamo", "NOMBRE_ENTIDAD"]]
bif_df = bif_df[bif_df['NOMBRE_ENTIDAD'].notna()] 
bif_df.to_excel("../../data/input_files/bancos_nombre_entidad.xlsx", encoding='latin-1', engine='xlsxwriter')
print(bif_df.head())