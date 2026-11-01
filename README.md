# Perfiles Ciudadanos de Participación Electoral (Reto 1 - Hackathon INE 2026)

Pipeline de analítica de datos, ingeniería de características longitudinales y segmentación no supervisada sobre las elecciones federales de México (2009, 2012, 2015, 2018, 2021 y 2024).

---

## 🏛️ Arquitectura del Repositorio

```text
procesamiento-datos/
├── data/
│   ├── clean/                        # Tablas procesadas y listas para modelado / BI
│   │   ├── base_limpia.parquet       # Conteo seccion x sexo x edad x anio (6.49M filas)
│   │   ├── base_municipio.parquet    # Conteo municipio x sexo x edad x anio (236k filas)
│   │   ├── catalogo_territorial.parquet # Nombres oficiales de municipios y entidades
│   │   ├── matriz_features.parquet   # 39,632 observaciones longitudinales con métricas
│   │   ├── perfiles_segmentados.parquet # Base completa con asignación de clusters
│   │   └── perfiles_resumen_municipio.csv / .parquet # Resumen para Power BI (2,477 municipios)
├── reports/
│   ├── calidad/                      # Validación con padrón nacional y tasas oficiales
│   └── clustering/                   # Métricas y visualizaciones de los modelos
│       ├── metricas_modelos.csv      # Comparación K-Means, GMM y Jerárquico (k=4,5,6)
│       ├── caracterizacion_perfiles.csv # Centroides y demografía de los 5 perfiles
│       ├── comparacion_metricas_k.png
│       ├── centroides_perfiles.png
│       └── composicion_edad_perfiles.png
├── src/
│   ├── config.py                     # Rutas, grupos de edad, constantes y entorno
│   ├── 01_ingest.py                  # Lectura de zips/csv crudos y tipificación
│   ├── 01b_modality.py               # Voto anticipado y prisión preventiva
│   ├── 02_cleaning.py                # Limpieza, agregación y filtros de calidad
│   ├── 03_bigquery_upload.py         # Carga de tablas verificadas a Google BigQuery
│   ├── 04_features.py                # Feature engineering longitudinal (2009-2024)
│   └── 05_clustering.py              # Modelado, validación ARI/Silueta y caracterización
├── requirements.txt
└── .env
```

---

## 🚀 Pipeline de Ejecución

1. **Ingesta y Limpieza:**
   ```bash
   python src/01_ingest.py
   python src/02_cleaning.py
   ```
2. **Feature Engineering Longitudinal:**
   ```bash
   python src/04_features.py
   ```
3. **Segmentación y Validación ($k \in \{4, 5, 6\}$):**
   ```bash
   python src/05_clustering.py
   ```
4. **Carga en BigQuery:**
   ```bash
   python src/03_bigquery_upload.py
   ```

---

## 👥 Resumen de los 5 Perfiles Ciudadanos

| Clave | Perfil Ciudadano | Part. Media | Volatilidad | Brecha Pres. vs Inter. | Tendencia (09→24) | % Lista Nominal | % Mujeres |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **P1** | **Abstencionismo Estructural y Desencanto** | 36.8% | 9.6% | +13.7% | +10.1% | 21.9% | 24.9% |
| **P2** | **Votantes Coyunturales (Presidenciales)** | 48.5% | 13.9% | **+21.6%** | **+22.8%** | 17.2% | 62.4% |
| **P3** | **Participación Intermedia Moderada** | 50.3% | 8.1% | +7.9% | +2.9% | **24.6%** | 47.1% |
| **P4** | **Ciudadanía Consistente en Ascenso** | 62.3% | 11.1% | +15.6% | +20.5% | 20.5% | **67.5%** |
| **P5** | **Núcleo Cívico de Alta Fidelidad** | **67.3%** | **6.8%** | **+5.7%** | +3.4% | 15.8% | 64.9% |
