#!/usr/bin/env python3
# -*- coding: utf-8 -*-
''' TAREA 1. CIENCIA DE DATOS. ALE, LUZ Y RODRI'''

import pandas as pd

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

df_data = df_data.rename(columns={'Site Code': 'Año'})


