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
#sns.boxplot(pd.to_numeric(df_data.iloc[:, 1:].values.flatten(),errors='coerce'),
#                          color="pink", fill=True)
#plt.show()

#===========================#===========================#===========================
#===========================Valores Faltantes==========================
#===========================#===========================#===========================

# Lista de valores que deben ser considerados como faltantes
VALORES_FALTANTES = [
    # Valores estándar
    np.nan, None, pd.NaT,
    
    # Strings comunes de missing
    '', ' ', '  ', '   ', '\t', '\n', '\r',
    'na', 'NA', 'N/A', 'n/a', 'NaN', 'nan',
    'null', 'NULL', 'None', 'none',
    'missing', 'MISSING', 'Unknown', 'unknown',
    '?', '??', '???', '-', '--', '---',
    
    # Valores numéricos que suelen usarse como missing
    -1, -999, -99, 999, 99, 9999,
    
    # Strings numéricos que suelen ser missing
    '-1', '-999', '-99', '999', '99', '9999'
]



def valores_faltantes(df):
    """
    Funcion para detectar valores faltantes y no afectar el codigo mientras los demas
    trabajan.
    """
    
    # Crear máscara de valores faltantes
    mask = pd.DataFrame(False, index=df.index, columns=df.columns)
    
    # Detectar por tipo de dato
    for col in df.columns:
        # Para columnas numéricas
        if pd.api.types.is_numeric_dtype(df[col]):
            # Valores numéricos que suelen ser missing
            mask[col] = df[col].isin(VALORES_FALTANTES)
        
        # Para columnas de texto
        elif df[col].dtype == 'object':
            # Convertir a string y detectar patrones de missing
            str_col = df[col].astype(str).str.strip().str.lower()
            
            # Patrones de texto que indican missing
            text_patterns = [
                'nan', 'na', 'n/a', 'null', 'none', 'missing', 
                'unknown', '?', '-', '', 'not available',
                'sin dato', 'no disponible', 'no especificado'
            ]
            
            for pattern in text_patterns:
                mask[col] = mask[col] | (str_col == pattern)

    # Detectar todos los valores faltantes
    mask_faltantes = mask
    
    # Calcular estadísticas
    missing_count = mask_faltantes.sum()
    missing_percent = (missing_count / len(df)) * 100
    total_missing = missing_count.sum()
    
    # Crear DataFrame con resultados
    missing_info = pd.DataFrame({
        'Columna': missing_count.index,
        'Valores_Faltantes': missing_count.values,
        'Porcentaje_Faltante': missing_percent.values
    })
    
    # Filtrar solo columnas con valores faltantes
    missing_info = missing_info[missing_info['Valores_Faltantes'] > 0]
    missing_info = missing_info.sort_values('Porcentaje_Faltante', ascending=False)
    
    if len(missing_info) > 0:        
        for _, row in missing_info.iterrows():
            col = row['Columna']
            count = row['Valores_Faltantes']
            percent = row['Porcentaje_Faltante']
            
            print(f"  {col}: {count} valores faltantes ({percent:.1f}%)")
                    
        # Gráfico
        plt.figure(figsize=(12, 8))
        bars = plt.bar(missing_info['Columna'], missing_info['Porcentaje_Faltante'], 
                      color='red', alpha=0.7)
        plt.title('Porcentaje de Valores Faltantes por Columna', 
                 fontsize=14, fontweight='bold')
        plt.ylabel('Porcentaje Faltante (%)', fontsize=12)
        plt.xlabel('Columnas', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # Añadir valores en las barras
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig("Faltantes.jpg", dpi=300)
        plt.show()
        
        print(f"\n TOTAL: {total_missing} valores faltantes en todo el dataset")
#        print(f" {total_missing/(len(df)*len(df.columns))*100:.2f}% de todas las celdas")
        
    else:
        print("No hay valores faltantes en ninguna columna")
    
    return missing_info

#Ejecucucion
valores_faltantes(df_data)


#===========Posibles Outliers===================
def convertir_columnas_numericas(df):
    """
    Funcion para convertir columnas que deberían ser numéricas pero estan como objetos
    """
    
    conversions_made = 0
    for col in df.columns:
        # Si ya es numérica, saltar
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
            
        # Intentar convertir a numérico
        try:
            # Primero limpiar strings: quitar espacios, comas, etc.
            if df[col].dtype == 'object':
                # Guardar valores originales para comparación
                original_non_null = df[col].dropna().shape[0]
                
                # Intentar conversión directa
                converted = pd.to_numeric(df[col], errors='coerce')
                
                # Verificar si la conversión fue exitosa para la mayoría de valores
                converted_non_null = converted.dropna().shape[0]
                
                if converted_non_null > 0:# and (converted_non_null / original_non_null) > 0.7:
                    df[col] = converted
#                    print(f"  ✓ Convertida: {col} -> numérica") #Para comprobar
                    conversions_made += 1
                    
        except Exception as e:
            print(f" Error convirtiendo {col}: {e}")
    
    if conversions_made == 0:
        print("  No se necesitaron conversiones")
    
    return df

#==========Columnas numericas======================
df_dataNum=convertir_columnas_numericas(df_data)

#Posibles Outliers, por RIQ y Z-score
def Outliers(df):
    """
    Mini funcion para detectar outliers
    """
    
    # Hacer una copia para no modificar el original
    df_clean = df.copy()
    
    
    # Seleccionar solo columnas numéricas
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) > 0:
        outliers_info = []
        
        print("Método 1: Rango Intercuartílico (IQR)")
        print("-" * 30)
        
        for col in numeric_cols:
            # Solo analizar columnas con suficientes datos
            if df_clean[col].notna().sum() > 10:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                
                if IQR > 0:  # Evitar división por cero
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = df_clean[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)][col]
                    outlier_percent = (len(outliers) / df_clean[col].notna().sum()) * 100
                    
                    if len(outliers) > 0:
                        print(f"  {col}: {len(outliers)} outliers ({outlier_percent:.1f}%)")
                        outliers_info.append({
                            'Columna': col,
                            'Outliers_IQR': len(outliers),
                            'Porcentaje_IQR': outlier_percent
                        })
        
        print("\nMétodo 2: Puntajes Z (Z-Scores > 3)")
        print("-" * 30)
        
        for col in numeric_cols:
            if df_clean[col].notna().sum() > 10:
                z_scores = np.abs(stats.zscore(df_clean[col].dropna()))
                outliers = z_scores > 3
                outlier_count = np.sum(outliers)
                outlier_percent = (outlier_count / len(z_scores)) * 100
                
                if outlier_count > 0:
                    print(f"  {col}: {outlier_count} outliers ({outlier_percent:.1f}%)")
        
        # Gráficos de boxplot para outliers
        n_cols = min(6, len(numeric_cols))
        if n_cols > 0:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            
            for i, col in enumerate(numeric_cols[1:7]):
                if i < len(axes):
                    df_clean[col].dropna().plot(kind='box', ax=axes[i])
                    axes[i].set_title(f'Boxplot de {col}')
                    axes[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
        
    else:
        print("  No hay columnas numéricas para análisis de outliers")
    return df_clean

#Ejecucucion de los Outliers
Outliers(df_dataNum)


#===========Ambiguedades=================
def ambiguedades(df):
    #INCONSISTENCIAS Y CODIFICACIÓN AMBIGUA

    # Hacer una copia para no modificar el original
    df_clean = df.copy()
    
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    
    if len(categorical_cols) > 0:
        print("Análisis de columnas categóricas:")
        for col in categorical_cols:
            unique_values = df_clean[col].dropna().unique()
            if len(unique_values) > 0:
                print(f"\n  {col}:")
                print(f"    Valores únicos: {len(unique_values)}")
                
                # Mostrar solo los primeros 10 valores únicos si hay muchos
#                if len(unique_values) <= 10:
#                    print(f"    Valores: {sorted(unique_values)}")
#                else:
#                    print(f"    Valores (primeros 10): {sorted(unique_values)[:10]}")
                
                # Detectar posibles duplicados por diferencias de formato
                lower_values = [str(x).lower().strip() for x in unique_values if pd.notna(x)]
                unique_lower = set(lower_values)
                
                if len(unique_lower) != len(unique_values):
                    print(f"  Posibles duplicados por diferencias de formato/capitalización")
    
    # Verificar consistencia en tipos de datos por columna
    print("\n Consistencia de tipos de datos:")
    type_issues = False
    for col in df_clean.columns:
        # Para columnas no numéricas, verificar mixed types
        if not pd.api.types.is_numeric_dtype(df_clean[col]):
            sample_values = df_clean[col].dropna().head(20)
            if len(sample_values) > 0:
                types_count = sample_values.apply(type).nunique()
                if types_count > 1:
                    print(f"  {col}: Múltiples tipos de datos encontrados")
                    type_issues = True
    
    if not type_issues:
        print("  Todos los tipos de datos son consistentes")

    return 
#================Ejecucion de ambiguedades para todo el data
ambiguedades(df)

