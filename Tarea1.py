#!/usr/bin/env python3
# -*- coding: utf-8 -*-
''' TAREA 1. CIENCIA DE DATOS. ALE, LUZ Y RODRI'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

#============Exploración de los datos===============

#Ajustamos la ruta al archivo de datos .csv:
file_path = "/Users/aleborrego/Downloads/Tarea 1 Ciencia de Datos/data.csv"

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

#Cambiamos el nombre de la columna 'Site Code' para que sea más claro que se trata del año de la medición
df_data = df_data.rename(columns={'Site Code': 'Año'})

#Resumen breve de los datos
df_data.describe()

#Gráfica de dispersión de todos los datos (sin distinción)
plt.figure(figsize=(10, 6))

for col in df_data.columns[1:]:
    plt.scatter(df_data['Año'], pd.to_numeric(df_data[col], errors='coerce'),
             marker='o', color = 'pink')
    
plt.xlabel('Año')
plt.ylabel('Valor')
plt.title('Observaciones vs Año')
plt.show()

##Gráfica de dispersión de todos los datos (con distinción por sitio)
plt.figure(figsize=(10, 6))

for col in df_data.columns[1:]:
    plt.scatter(df_data['Año'], pd.to_numeric(df_data[col], errors='coerce'),
             marker='o')
    
plt.xlabel('Año')
plt.ylabel('Valor')
plt.title('Observaciones vs Año')
plt.show()

#Gráfica de dispersión para distinguir y analizar menos sitios
plt.figure(figsize=(10, 6))

for col in df_data.columns[1:6]:
    plt.scatter(df_data['Año'], pd.to_numeric(df_data[col], errors='coerce'),
             marker='o', label = col)

plt.xlabel('Año')
plt.ylabel('Valor')
plt.title('Todas las variables vs Año')
plt.legend()
plt.show()

#Más gráficas considerando todos los datos

#Histograma
sns.histplot(pd.to_numeric(df_data.iloc[:, 1:].values.flatten(),
                           errors='coerce'), kde=False, color="pink")
plt.show()
    
#Densidad kernel
sns.kdeplot(pd.to_numeric(df_data.iloc[:, 1:].values.flatten(), errors='coerce'),
                          color="pink", fill=True)
plt.show()
    
#Boxplot
sns.boxplot(pd.to_numeric(df_data.iloc[:, 1:].values.flatten(),errors='coerce'),
                          color="pink", fill=True)
plt.show()


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

#a

#===========


