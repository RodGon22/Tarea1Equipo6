#!/usr/bin/env python3
# -*- coding: utf-8 -*-
''' TAREA 1. CIENCIA DE DATOS. ALE, LUZ Y RODRI'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


#Ajustamos la ruta al archivo de datos .csv:
file_path = "/Users/rodri/Downloads/data.csv"

#Creamos un DataFrame con las características en filas y las muestras en columnas.
df = pd.read_csv(file_path)

#Separamos del DataFrame las filas que contienen la info de los datos
df_info = df.drop(df.index[8:415])
#Calculamos el número de años en los que se tomaron registros de datos por sitio 
df_info.loc['Años registrados'] = (1 + df_info.iloc[6, 1:].astype(int) - df_info.iloc[5, 1:].astype(int))

#Separamos del DataFrame las columnas que contienen las muestras
df_data = df.drop(df.index[0:9])
#Recorremos el índice para que inicie en 0
df_data = df_data.reset_index(drop=True)

df_data = df_data.rename(columns={'Site Code': 'Año'})

#============Valores Faltantes===============

# Porcentaje de valores faltantes por columna
porcentaje_faltantes = (df_data.isnull().sum() / len(df)) * 100
porcentaje_faltantes = porcentaje_faltantes[porcentaje_faltantes > 0].sort_values(ascending=False)

print("\nPorcentaje de valores faltantes por variable:")
for col, percentage in porcentaje_faltantes.items():
    print(f"{col}: {percentage:.2f}%")

#Grafica ilustrativa
plt.figure(figsize=(12, 8))
faltante_data = df_data.isnull().sum()
faltante_data = faltante_data[faltante_data > 0]
faltante_data.sort_values(ascending=True).plot(kind='barh',color='red')
plt.title('Cantidad de valores faltantes por variable')
plt.xlabel('Número de valores faltantes')
plt.tight_layout()
#plt.savefig('valores_faltantes.png', dpi=300)
plt.show()


#===========


