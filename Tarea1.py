''' 
TAREA 1. 
INTRODUCCION A CIENCIA DE DATOS. 
ALE BL, LUZ SM Y RODRI GS
12 de septiembre del 2025
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

#============Exploración de los datos===============

#Ajustamos la ruta al archivo de datos .csv:
#file_path = "/Users/aleborrego/Downloads/Tarea 1 Ciencia de Datos/data.csv"
file_path = "/Users/rodri/Downloads/data.csv"

#Creamos un DataFrame con las características en filas y las muestras en columnas.
df = pd.read_csv(file_path)

#Separamos del DataFrame las filas que contienen la info de los datos y la transponemos
df_info = df.drop(df.index[8:415])
#Calculamos el número de años en los que se tomaron registros de datos por sitio 
df_info.loc['Años registrados'] = (1 + df_info.iloc[6, 1:].astype(int) - df_info.iloc[5, 1:].astype(int))

#Separamos del DataFrame las columnas que contienen las muestras
df_data = df.drop(df.index[0:9])
#Recorremos el índice para que inicie en 0
df_data = df_data.reset_index(drop=True)

#Cambiamos el nombre de la columna 'Site Code' para que sea más claro que se trata del año de la medición
df_data = df_data.rename(columns={'Site Code': 'Año'})

#Resumen estadístico de los datos
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

#Asegura que el año es numérico y crece de forma monótona
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

# 1)Otra  faltantes

#indexar por Año
df_t = df_data.copy()
if df_t.columns[0].lower() in ["año", "ano", "year"]:
    df_t = df_t.rename(columns={df_t.columns[0]: "Año"})
df_t = df_t.set_index("Año").sort_index()

plt.figure(figsize=(12, 4))
sns.heatmap(df_t.isna(), cbar=False)
plt.title("Mapa de faltantes por sitio y año (NaN)")
plt.xlabel("Sitio"); plt.ylabel("Año")
plt.tight_layout(); plt.show()


#===========Posibles Outliers===================
#La siguiente funcion se volvi+o innecesaria
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

#Ejecucion de los Outliers
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

# Opcional (solo para el mapa de faltantes al inicio/fin):

# Modelos de espacio de estados:
from statsmodels.tsa.statespace.structural import UnobservedComponents
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)


# 1) Estrategia 1: Eliminación por período común (sin AHI)
#    - Excluye AHI del cálculo del período común
#    - Fuerza a empezar en 1900

START_YEAR = 1900
EXCLUDE_FOR_COMMON = ["AHI"]

def periodo_comun(df, start_year=START_YEAR, exclude_cols=None):
    cols = df.columns.tolist()
    if exclude_cols:
        cols = [c for c in cols if c not in set(exclude_cols)]
    # recorta a partir de start_year
    df_recent = df.loc[df.index >= start_year, cols]
    # primer y último año válido por columna
    first_valid = df_recent.apply(lambda s: s.first_valid_index())
    last_valid  = df_recent.apply(lambda s: s.last_valid_index())
    # inicio = máximo de inicios; fin = mínimo de finales
    start = first_valid.max()
    end   = last_valid.min()
    if pd.isna(start) or pd.isna(end) or start > end:
        return None, None, df_recent.columns.tolist()
    return int(start), int(end), df_recent.columns.tolist()

anio_ini, anio_fin, cols_common = periodo_comun(df_t, START_YEAR, EXCLUDE_FOR_COMMON)
print(f"Período común SIN {EXCLUDE_FOR_COMMON}, desde {START_YEAR}: {anio_ini}–{anio_fin}")

if anio_ini is None:
    df_common = pd.DataFrame(index=df_t.index, columns=cols_common)
else:
    df_common = df_t.loc[anio_ini:anio_fin, cols_common].copy()
    # Por seguridad elimina filas con cualquier NaN (no debería haber si el rango es correcto)
    df_common = df_common.dropna(how="any")


# 2) Estrategia 2: Imputación temporal (Kalman) por sitio
#    - Aplica a todos los sitios, incluido AHI

def imputar_estado_espacio(serie, show_warnings=False):
    if serie.notna().sum() == 0:
        return serie.copy(), None
    i0, i1 = serie.first_valid_index(), serie.last_valid_index()
    sub = serie.loc[i0:i1].astype(float)
    try:
        mod = UnobservedComponents(
            sub,
            level="local linear trend",
            autoregressive=1,
            mle_regression=False
        )
        res = mod.fit(disp=False, maxiter=200)
        pred = res.predict()  # media suavizada
    except Exception as e:
        if show_warnings:
            print(f"[{serie.name}] UCM falló; uso interpolación lineal. Motivo: {e}")
        pred = sub.interpolate(limit_direction="both")

    imputada = serie.copy()
    na_mask = sub.isna()
    imputada.loc[sub.index[na_mask]] = pred.loc[na_mask]
    # Fuera de [i0,i1] (bordes) se deja NaN para evitar extrapolaciones largas
    return imputada, None

df_kalman = pd.DataFrame(index=df_t.index, columns=df_t.columns, dtype=float)
for col in df_t.columns:
    imp_col, _ = imputar_estado_espacio(df_t[col])
    df_kalman[col] = imp_col

# -----------------------------------------------------------
# 3) Comparación visual 
#    - Histogramas: Original vs Eliminación (sin AHI) vs Kalman
# -----------------------------------------------------------
cols_sitios = list(df_t.columns)
ncols = 4
nrows = 2
nplots = min(len(cols_sitios), ncols)

fig, axes = plt.subplots(nrows, ncols, figsize=(18, 8))
axes = axes.flatten()

for i, col in enumerate(cols_sitios[:nplots]):
    orig = df_t[col].dropna()
    elim_vals = df_common[col].dropna() if (col in df_common.columns) else pd.Series(dtype=float)
    kalm_vals = df_kalman[col].dropna()

    # Arriba: Original vs Eliminación (sin AHI, >=1900)
    ax = axes[i]
    if len(orig) > 0:
        ax.hist(orig, bins=30, alpha=0.5, label="Original")
    if len(elim_vals) > 0:
        ax.hist(elim_vals, bins=30, alpha=0.5, label="Eliminación (común ≥1900, sin AHI)")
    ax.set_title(f"{col}: Original vs Eliminación")
    ax.legend(fontsize=8)

    # Abajo: Original vs Kalman (toda la serie)
    ax2 = axes[i + nplots]
    if len(orig) > 0:
        ax2.hist(orig, bins=30, alpha=0.5, label="Original")
    if len(kalm_vals) > 0:
        ax2.hist(kalm_vals, bins=30, alpha=0.5, label="Imputación Kalman")
    ax2.set_title(f"{col}: Original vs Kalman")
    ax2.legend(fontsize=8)

plt.tight_layout(); plt.show()

# -----------------------------------------------------------
# 4) Series temporales de ejemplo (overlay)
# -----------------------------------------------------------
sitios_demo = cols_sitios[:3]  # elige 3–4 para ilustrar
for col in sitios_demo:
    plt.figure(figsize=(11,4))
    plt.plot(df_t.index, df_t[col], marker="o", lw=1, label="Original (obs.)")
    if col in df_common.columns and not df_common[col].empty:
        plt.plot(df_common.index, df_common[col], lw=2, label="Eliminación (≥1900, sin AHI)")
    plt.plot(df_kalman.index, df_kalman[col], lw=1.75, label="Imputación Kalman")
    plt.title(f"Serie {col}: Original vs Eliminación vs Kalman")
    plt.xlabel("Año"); plt.ylabel("δ13C VPDB (‰)")
    plt.legend(); plt.tight_layout(); plt.show()

# -----------------------------------------------------------
# 5) Métricas para el reporte
# -----------------------------------------------------------
total_obs = df_t.size
total_obs_no_nan = df_t.notna().sum().sum()
elim_obs = df_common.size if not df_common.empty else 0
print(f"Observaciones no NaN (original): {total_obs_no_nan}/{total_obs}")
if not df_common.empty:
    print(f"Observaciones tras eliminación (≥{START_YEAR}, sin {EXCLUDE_FOR_COMMON}): "
          f"{elim_obs} ({elim_obs/total_obs:.1%} del total)")
else:
    print("No existe un período común sin faltantes bajo las restricciones (≥1900 y sin AHI).")

###############============================
#======================================Visualización con imputacion de Kalman
#======================================Relación entre dos variables
#======================================Posibles Nuevos Outliers por la imputación


def visualizaciones_exploratorias(df_original, df_imputado, sitio_ejemplo):
    """    
    Parameters:
    df_original: DataFrame con datos originales (con NaN)
    df_imputado: DataFrame con datos imputados (sin NaN)
    sitio_ejemplo: Sitio específico para analizar
    """
    
    # Asegurar que el índice sea numérico (años)
    if not pd.api.types.is_numeric_dtype(df_original.index):
        df_original = df_original.reset_index()
        if 'Año' in df_original.columns:
            df_original = df_original.set_index('Año')
    
    if not pd.api.types.is_numeric_dtype(df_imputado.index):
        df_imputado = df_imputado.reset_index()
        if 'Año' in df_imputado.columns:
            df_imputado = df_imputado.set_index('Año')
    
    #Distribucion de variable continua Normal
    plt.figure(figsize=(12, 4))
    
    # Datos originales vs imputados
    datos_originales = df_original[sitio_ejemplo].dropna()
    datos_imputados = df_imputado[sitio_ejemplo] #solo elegimos uno para comparar
    
    plt.subplot(1, 2, 1)
    sns.histplot(datos_originales, kde=True, color='skyblue', 
                 alpha=0.7, label='Original', stat='density')
    sns.histplot(datos_imputados, kde=True, color='coral', 
                 alpha=0.7, label='Imputado', stat='density')
    plt.title(f'Distribución de {sitio_ejemplo}\nOriginal vs Imputado')
    plt.xlabel('δ¹³C (‰ VPDB)')
    plt.ylabel('Densidad')
    plt.legend()
    
    # QQ-plot para normalidad
    plt.subplot(1, 2, 2)
    stats.probplot(datos_imputados.dropna(), dist="norm", plot=plt)
    plt.title(f'QQ-Plot de {sitio_ejemplo}\n(Test de normalidad)')
    plt.tight_layout()
#    plt.savefig("QQplotREN.jpg",dpi=300)
    plt.show()
        
    #  Relacion lineal entre la variable de interés con la de mayor correlacion
    # (hay que añadir datos impitados)
    correlaciones = df_imputado.corr().abs()
    sitio1 = sitio_ejemplo
    
    # Elegir sitio con más correlacionado
    correlaciones_sitio = correlaciones[sitio1].drop(sitio1)
    sitio2 = correlaciones_sitio.idxmax()
    
    # FILTRAR DATOS
    mask_valid = df_imputado[sitio1].notna() & df_imputado[sitio2].notna()
    df_clean = df_imputado.loc[mask_valid, [sitio1, sitio2]]
    
    plt.figure(figsize=(10, 6))
    
    # Datos originales (puntos sólidos)
    mask_original = df_original[sitio1].notna() & df_original[sitio2].notna()
    plt.scatter(df_original.loc[mask_original, sitio1], 
                df_original.loc[mask_original, sitio2], 
                alpha=0.7, color='blue', label='Datos originales', s=30)
    
    # Datos imputados (puntos transparentes)
    plt.scatter(df_imputado[sitio1], df_imputado[sitio2], 
                alpha=0.3, color='red', label='Datos imputados', s=20)
    
    # Línea de tendencia
    if len(df_clean) > 1:  # Se necesita al menos 2 puntos para ajustar, y cachar errores
        try:
            z = np.polyfit(df_clean[sitio1], df_clean[sitio2], 1)
            p = np.poly1d(z)
            
            # Generar puntos para la línea de tendencia
            x_range = np.linspace(df_clean[sitio1].min(), df_clean[sitio1].max(), 100)
            plt.plot(x_range, p(x_range), color='black', linestyle='--', alpha=0.8, label='Tendencia lineal')
            
        except (LinAlgError, ValueError) as e:
            print(f"Advertencia: No se pudo ajustar línea de tendencia: {e}")
            # Mostrar mensaje en el gráfico
            plt.text(0.5, 0.9, 'No se pudo calcular tendencia lineal', 
                    transform=plt.gca().transAxes, ha='center', color='red')
    else:
        print("Adv: No hay suficientes datos")
    
    plt.xlabel(f'{sitio1} (δ¹³C ‰)')
    plt.ylabel(f'{sitio2} (δ¹³C ‰)')
    plt.title(f'Relación entre {sitio1} y {sitio2}\n(Coef. correlación: {correlaciones_sitio.max():.3f})')
    plt.legend()
    plt.grid(alpha=0.3)
#    plt.savefig("MasCorreREN.jpg",dpi=300)
    plt.show()
        
    # Nuevos posibles Outliers Y V.E.
    plt.figure(figsize=(15, 5))
    
    # Boxplot por sitio (primeros 6 sitios)
    sitios_analizar = df_imputado.columns[:6]  # Primeros 6 sitios
    
    # Filtrar solo columnas con datos
    sitios_analizar = [col for col in sitios_analizar if df_imputado[col].notna().sum() > 0]
    
    plt.subplot(1, 2, 1)
    data_plot = [df_imputado[col].dropna() for col in sitios_analizar]
    plt.boxplot(data_plot, labels=sitios_analizar)
    plt.xticks(rotation=45)
    plt.title('Boxplot por sitio (detectando outliers)')
    plt.ylabel('δ¹³C (‰ VPDB)')
    
    # Z-scores para detectar outliers extremos
    plt.subplot(1, 2, 2)
    
    # Calcular outliers por sitio individualmente
    outlier_counts = []
    for col in sitios_analizar:
        col_data = df_imputado[col].dropna()
        if len(col_data) > 0:
            z_scores = np.abs(stats.zscore(col_data))
            outlier_count = (z_scores > 3).sum()
        else:
            outlier_count = 0
        outlier_counts.append(outlier_count)
    
    plt.bar(sitios_analizar, outlier_counts, color='orange', alpha=0.7)
    plt.xticks(rotation=45)
    plt.title('Número de outliers por sitio (Z-score > 3)')
    plt.ylabel('Cantidad de outliers')
    
    # Añadir valores en las barras
    for i, count in enumerate(outlier_counts):
        plt.text(i, count + 0.1, str(count), ha='center', va='bottom')
    
    plt.tight_layout()
#    plt.savefig("OutliersREN.jpg",dpi=300)
    plt.show()
        
    # Estadísticas adicionales de outliers
    print("Resumen de outliers por sitio (Z-score > 3):")
    for sitio, count in zip(sitios_analizar, outlier_counts):
        total_datos = df_imputado[sitio].notna().sum()
        porcentaje = (count / total_datos) * 100 if total_datos > 0 else 0
        print(f"  {sitio}: {count} outliers ({porcentaje:.1f}%)")

# Ejecutar (usando df_t como original y df_kalman como imputado)

# Verificar que los DataFrames no estén vacíos
if df_kalman is not None and not df_kalman.empty:
    visualizaciones_exploratorias(df_t, df_kalman, sitio_ejemplo='REN ')
else:
    print("Error: df_kalman está vacío o no definido")
    print("Usando datos originales para visualización (sin imputación)")
    visualizaciones_exploratorias(df_t, df_t, sitio_ejemplo='REN ')




#=========================================================
#============ Codificación y escalamiento ===============

#Transformación de variables categóricas
df_infoT=df_info.T #Transponemos el date frame de la información
df_infoT.columns = df_infoT.iloc[0] #Asigna la primera fila como encabezados de las columna
info = df_infoT[1:].reset_index(drop=True)  # Eliminamos primera fila y reiniciamos el índice

#Aplicamos codificación one-hot para trabajar con var.cat. si se necesita. p.ej: sitios y especies
site_names_onehot = pd.get_dummies(df_infoT['Site  name']).astype(int)
species_onehot = pd.get_dummies(df_infoT['Species'].unique()).astype(int)

#Escalamiento de datos
#Creamos un data frame con las columnas CAZ y REN
df_caz_vin = df_data[['CAZ ', 'VIN ']].copy()
df_caz_vin.columns = ['CAZ', 'VIN']  #Quitamos los espacios de los nombres para evitar errores futuros

#Escalamos las columnas CAZ y REN  usando normalización min-max
min_caz = df_caz_vin['CAZ'].min()
max_caz = df_caz_vin['CAZ'].max()
df_caz_vin['CAZ min_max'] = (df_caz_vin['CAZ'] - min_caz) / (max_caz - min_caz)

min_vin = df_caz_vin['VIN'].min()
max_vin = df_caz_vin['VIN'].max()
df_caz_vin['VIN min_max'] = (df_caz_vin['VIN'] - min_vin) / (max_vin - min_vin)
    
#Escalamos las columnas CAZ y REN  usando estandarización z-score
mean_caz = df_caz_vin['CAZ'].mean()
stdv_caz = df_caz_vin['CAZ'].std()
df_caz_vin['CAZ z-score'] = (df_caz_vin['CAZ'] - mean_caz) / (stdv_caz)

mean_vin = df_caz_vin['VIN'].mean()
stdv_vin = df_caz_vin['VIN'].std()
df_caz_vin['VIN z-score'] = (df_caz_vin['VIN'] - mean_vin) / (stdv_vin)

    
# Grafica compuesta

#Box-plot para min-max
fig, axes = plt.subplots(1, 2, figsize=(16, 10))

sns.boxplot(pd.to_numeric(df_caz_vin.iloc[:, 2].values.flatten()),
                          color="pink", fill=True, ax = axes[0])
sns.boxplot(pd.to_numeric(df_caz_vin.iloc[:, 3].values.flatten()),
                          color="pink", fill=True, ax = axes[1])
plt.tight_layout()
plt.show()

#Box-plot para z-score
fig, axes = plt.subplots(1, 2, figsize=(16, 10))

sns.boxplot(pd.to_numeric(df_caz_vin.iloc[:, 4].values.flatten()),
                          color="pink", fill=True, ax = axes[0])
sns.boxplot(pd.to_numeric(df_caz_vin.iloc[:, 5].values.flatten()),
                          color="pink", fill=True, ax = axes[1])

plt.tight_layout()
plt.show()

#Histograma para min-max
fig, axes = plt.subplots(1, 2, figsize=(16, 10))

sns.histplot(pd.to_numeric(df_caz_vin.iloc[:, 2].values.flatten()),
                          color="pink", fill=True, ax = axes[0])
sns.histplot(pd.to_numeric(df_caz_vin.iloc[:, 3].values.flatten()),
                          color="pink", fill=True, ax = axes[1])

plt.tight_layout()
plt.show()

#Histograma para z-score
fig, axes = plt.subplots(1, 2, figsize=(16, 10))

sns.histplot(pd.to_numeric(df_caz_vin.iloc[:, 4].values.flatten()),
                          color="pink", fill=True, ax = axes[0])
sns.histplot(pd.to_numeric(df_caz_vin.iloc[:, 5].values.flatten()),
                          color="pink", fill=True, ax = axes[1])

plt.tight_layout()
plt.show()