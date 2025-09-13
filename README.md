# Análisis de Datos: Stable Carbon ISONET

## Descripción del Proyecto

Este repositorio contiene el análisis de datos del proyecto **ISONET** (400 Years of Annual Reconstructions of European Climate Variability Using a Highly Resolved Isotopic Network). El trabajo se centra en el estudio de las series temporales de la relación isotópica δ¹³C medida en celulosa de anillos de árboles de 24 sitios distribuidos por Europa, con datos que abarcan desde el año 1600 hasta 2003.

El objetivo principal fue realizar un proceso completo de **limpieza, exploración, manejo de datos faltantes y preparación** de los datos para su uso en modelos estadísticos. 

## Autores

- Luz María Salazar M.
- Rodrigo Gonzaga S.
- María Alejandra Borrego L.

**Profesor:** Dr. Marco Antonio Aquino López  
**Institución:** Centro de Investigación en Matemáticas (CIMAT) A.C.

## Resumen por sección

### 1. Detección de Problemas en los Datos
- **Datos Faltantes**
- **Valores Atípicos (Outliers):** Se utilizaron los métodos **IQR** y **Z-Score**.

### 2. Manejo y Mantenimiento de los Datos
-  **Eliminación por Período Común**
-  **Imputación Temporal con Modelo en Espacio de Estados + Suavizado de Kalman** 


### 3. Codificación y Escalamiento
- **One-Hot Encoding** 
- **Normalización Min-Max** y **Estandarización Z-Score**.

### 4. Visualización
- Mapas de calor de datos faltantes.
- Histogramas.
- Gráficos de series de tiempo antes y después de la imputación.
- Gráficos Q-Q para pruebas de normalidad.

## Tecnologías:

- **Lenguajes:** Python (Pandas, NumPy, SciPy, Statsmodels) y/o R.
- **Visualización:** Matplotlib, Seaborn, Plotly.
- **Control de Versiones:** Git / GitHub.
- **Documentación:** LaTeX .





**Nota:** Los datos originales de ISONET son propiedad de los autores Schleser, G. H., et al. (2023) y deben citarse apropiadamente. Este repositorio contiene solo el análisis realizado sobre ellos.
