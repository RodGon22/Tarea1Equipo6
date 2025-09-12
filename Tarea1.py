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

#Más gráficas considerando todos los datos

#Histograma
sns.histplot(pd.to_numeric(df_data.iloc[:, 1:].values.flatten(),
                           errors='coerce'), kde=False, color="pink")
plt.show()
    
#Densidad kernel
sns.kdeplot(pd.to_numeric(df_data.iloc[:, 1:].values.flatten(), errors='coerce'),
                          color="pink", fill=True)
plt.show()

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

#Acegura que el año es numérico y crece de forma monótona
df_data['Año'] = pd.to_numeric(df_data['Año'], errors='coerce').astype('Int64')
assert df_data['Año'].dropna().between(1500, 2100).all()
assert df_data['Año'].is_monotonic_increasing

# columnas de sitios deben ser numéricas
sitios = [c for c in df_data.columns if c != 'Año']
for c in sitios:
    df_data[c] = pd.to_numeric(df_data[c], errors='coerce')

# Gráfica de lineas con los 6 sitios con más datos
n_obs = df_data[sitios].notna().sum().sort_values(ascending=False)
top = list(n_obs.head(6).index)

plt.figure(figsize=(10,6))
for s in top:
    plt.plot(df_data['Año'], df_data[s], marker='.', linewidth=1, label=s, alpha=0.9)
plt.xlabel('Año'); plt.ylabel('δ13C (‰ vs VPDB)')
plt.title('δ13C por año — sitios con mayor cobertura')
plt.legend(); plt.grid(alpha=0.3); plt.show()

#Gráfica de dispersión para distinguir y analizar menos sitios
#plt.figure(figsize=(10, 6))

#for col in df_data.columns[1:6]:
#    plt.scatter(df_data['Año'], pd.to_numeric(df_data[col], errors='coerce'),
#             marker='o', label = col)

#plt.xlabel('Año')
#plt.ylabel('Valor')
#plt.title('Todas las variables vs Año')
#plt.legend()
#plt.show()


    
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

#=================  MANEJO DE DATOS FALTANTES ==========================
# =========================================================
# preparación (normaliza nombres de columnas)

def normalizar_nombres(df: pd.DataFrame) -> pd.DataFrame:
    copia = df.copy()
    copia.columns = (copia.columns
                     .str.replace(r"\s+", " ", regex=True)
                     .str.strip()
                     .str.upper())
    return copia

def _detectar_columnas(df: pd.DataFrame, col_anio: str = "AÑO"):
    cols = list(df.columns)
    if col_anio not in cols and "YEAR CE" in cols:
        col_anio = "YEAR CE"
    if col_anio not in df.columns:
        col_anio = df.columns[0]
    sitios = [c for c in df.columns if c != col_anio]
    return col_anio, sitios

def _forzar_numerico(df: pd.DataFrame, columnas):
    copia = df.copy()
    for c in columnas:
        copia[c] = pd.to_numeric(copia[c], errors="coerce")
    return copia

# ============================================
# 1) Estrategia 1 — Interpolación lineal simple
# ============================================
def imputar_interpolacion(df: pd.DataFrame, col_anio: str = "AÑO", limit_area: str = "inside"):
    col_anio, sitios = _detectar_columnas(df, col_anio)
    base = _forzar_numerico(df, [col_anio] + sitios).sort_values(col_anio).reset_index(drop=True)
    na_inicial = base[sitios].isna()

    salida = base.copy()
    for s in sitios:
        salida[s] = salida[s].interpolate(method="linear", limit_area=limit_area)

    mascara = na_inicial & salida[sitios].notna()
    return salida, mascara

# ==============================================================
# 2) Estrategia B — Máxima verosimilitud Normal (regresión t,t^2)
# ==============================================================
def imputar_mle_normal(df: pd.DataFrame, col_anio: str = "AÑO"):
    col_anio, sitios = _detectar_columnas(df, col_anio)
    base = _forzar_numerico(df, [col_anio] + sitios).sort_values(col_anio).reset_index(drop=True)

    t = base[col_anio].astype(float).to_numpy()
    X = np.c_[np.ones_like(t), t, t**2]  # diseño cuadrático

    salida = base.copy()
    na_inicial = base[sitios].isna()

    for s in sitios:
        y = salida[s].to_numpy(dtype=float)
        obs = ~np.isnan(y)
        if obs.sum() < 3:
            continue
        beta, *_ = np.linalg.lstsq(X[obs], y[obs], rcond=None)
        yhat = X @ beta
        y[~obs] = yhat[~obs]   # media MLE
        salida[s] = y

    mascara = na_inicial & salida[sitios].notna()
    return salida, mascara

# =================================
# 3) Métrica y visualizaciones 
# =================================
def porcentaje_imputado(mascara_imp: pd.DataFrame) -> pd.Series:
    return (mascara_imp.sum() / mascara_imp.shape[0] * 100).sort_values(ascending=False)

def histogramas_comparativos(df_original: pd.DataFrame,
                             df_A: pd.DataFrame, df_B: pd.DataFrame,
                             sitios: list, col_anio: str = "AÑO",
                             etiqueta_A: str = "Interpolación",
                             etiqueta_B: str = "MLE Normal",
                             bins: int = 30):
    """
    Para cada sitio seleccionado dibuja:
      - Fila 1: Original vs Interpolación
      - Fila 2: Original vs MLE Normal
    """
    n = len(sitios)
    ncols = min(4, n)
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5*ncols, 7), squeeze=False)

    for i, sitio in enumerate(sitios[:ncols]):
        # Original vs Interpolación
        ax = axes[0, i]
        ax.hist(pd.to_numeric(df_original[sitio], errors="coerce").dropna(), bins=bins, alpha=0.5, label="Original")
        ax.hist(pd.to_numeric(df_A[sitio], errors="coerce"), bins=bins, alpha=0.5, label=etiqueta_A)
        ax.set_title(f"{sitio} (Original vs {etiqueta_A})", fontsize=10)
        ax.legend(fontsize=8)

        # Original vs MLE
        ax2 = axes[1, i]
        ax2.hist(pd.to_numeric(df_original[sitio], errors="coerce").dropna(), bins=bins, alpha=0.5, label="Original")
        ax2.hist(pd.to_numeric(df_B[sitio], errors="coerce"), bins=bins, alpha=0.5, label=etiqueta_B)
        ax2.set_title(f"{sitio} (Original vs {etiqueta_B})", fontsize=10)
        ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.show()

def barras_porcentaje_imputado(pct_A: pd.Series, pct_B: pd.Series,
                               titulo_A="Interpolación", titulo_B="MLE Normal"):
    """Barras comparativas del % imputado por sitio para cada estrategia (top 12)."""
    topA = pct_A.head(12)
    topB = pct_B.head(12)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
    axes[0].bar(topA.index, topA.values, alpha=0.8)
    axes[0].set_title(f"% imputado por sitio — {titulo_A}")
    axes[0].set_ylabel("%")
    axes[0].tick_params(axis='x', rotation=45)

    axes[1].bar(topB.index, topB.values, alpha=0.8, color="tab:orange")
    axes[1].set_title(f"% imputado por sitio — {titulo_B}")
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()



# Normaliza nombres
df_norm = normalizar_nombres(df_data)
col_anio, sitios = _detectar_columnas(df_norm, "AÑO")

#  Aplica cada estrategia 
df_interp, mascara_interp = imputar_interpolacion(df_norm, col_anio="AÑO", limit_area="inside")
df_mle,    mascara_mle    = imputar_mle_normal(df_norm,    col_anio="AÑO")

#  % imputado por sitio (para tablas/figuras)
pct_interp = porcentaje_imputado(mascara_interp).round(1)
pct_mle    = porcentaje_imputado(mascara_mle).round(1)
print("Interpolación (% imputado):\n", pct_interp)
print("\nMLE Normal   (% imputado):\n", pct_mle)

barras_porcentaje_imputado(pct_interp, pct_mle)

#  Selección de sitios a graficar (los mejor cubiertos para que los histogramas sean informativos)
conteo_no_na = df_norm[sitios].notna().sum().sort_values(ascending=False)
sitios_top = conteo_no_na.head(4).index.tolist()  # cambia 4 por el número que quieras mostrar

# Histogramas comparativos (tipo profe, pero con nuestras estrategias)
histogramas_comparativos(df_norm, df_interp, df_mle, sitios_top, col_anio="AÑO")

#   Histograma global (todas las columnas apiladas)
def histograma_global(df, titulo, bins=40):
    vals = pd.to_numeric(df[[c for c in df.columns if c not in ("AÑO","YEAR CE")]].values.ravel(), errors="coerce")
    vals = pd.Series(vals).dropna()
    plt.figure(figsize=(6,4))
    plt.hist(vals, bins=bins, alpha=0.8)
    plt.title(titulo); plt.xlabel("δ13C (‰ vs VPDB)"); plt.ylabel("Frecuencia")
    plt.tight_layout(); plt.show()

histograma_global(df_norm, "Original (todas las series)")
histograma_global(df_interp, "Imputación por Interpolación (todas las series)")
histograma_global(df_mle,    "Imputación por MLE Normal (todas las series)")

