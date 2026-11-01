"""Paso 4: Ingeniería de Características (Feature Engineering) para Segmentación.
Construye la matriz longitudinal por unidad demográfico-territorial:
  (entidad_cve, municipio_cve, sexo, grupo_edad)
Calcula tasas de participación por elección, media histórica, volatilidad,
brecha presidencial vs intermedia, tendencia temporal y peso de lista nominal.
Salida: data/clean/matriz_features.parquet
"""
from pathlib import Path
import numpy as np
import pandas as pd

from config import (
    CLEAN,
    CALIDAD,
    ANIOS,
    PRESIDENCIALES,
    INTERMEDIAS,
    SEXO_MAP,
)


def construir_features():
    print("Cargando base municipal y catálogo...")
    df_mun = pd.read_parquet(CLEAN / "base_municipio.parquet")
    df_cat = pd.read_parquet(CLEAN / "catalogo_territorial.parquet")

    # Mapeo de nombres más recientes de municipio y entidad
    nombres = (
        df_cat.sort_values("anio")
        .groupby(["entidad_cve", "municipio_cve"])[["entidad", "municipio"]]
        .last()
        .reset_index()
    )

    # Filtrar únicamente sexos con serie histórica continua (0: Hombres, 1: Mujeres)
    # Sexo 2 (No binario) solo existe a partir de 2024 con N=105 a nivel nacional
    df_analisis = df_mun[df_mun["sexo"].isin([0, 1])].copy()
    df_analisis["sexo_desc"] = df_analisis["sexo"].map(SEXO_MAP).astype("string")

    # Calcular tasa de participación por fila
    df_analisis["tasa_part"] = np.where(
        df_analisis["ln"] > 0,
        df_analisis["sv"] / df_analisis["ln"],
        np.nan,
    )
    # Acotar tasas posibles por inconsistencias o redondeos entre 0 y 1
    df_analisis["tasa_part"] = df_analisis["tasa_part"].clip(0.0, 1.0)

    # Pivotear las tasas de participación por año
    piv_part = df_analisis.pivot_table(
        index=["entidad_cve", "municipio_cve", "sexo", "grupo_edad"],
        columns="anio",
        values="tasa_part",
        aggfunc="mean",
    )
    piv_part.columns = [f"part_{col}" for col in piv_part.columns]

    # Pivotear la lista nominal para tener el peso poblacional
    piv_ln = df_analisis.pivot_table(
        index=["entidad_cve", "municipio_cve", "sexo", "grupo_edad"],
        columns="anio",
        values="ln",
        aggfunc="sum",
    )
    piv_ln.columns = [f"ln_{col}" for col in piv_ln.columns]

    # Unir ambas tablas
    features = piv_part.join(piv_ln).reset_index()

    # Columnas de años esperadas
    cols_part = [f"part_{a}" for a in ANIOS]
    cols_pres = [f"part_{a}" for a in PRESIDENCIALES]
    cols_inter = [f"part_{a}" for a in INTERMEDIAS]
    cols_ln = [f"ln_{a}" for a in ANIOS]

    # Imputar municipios nuevos con la mediana estatal para ese mismo sexo y grupo de edad
    for col in cols_part:
        if col not in features.columns:
            features[col] = np.nan
        # Mediana por entidad, sexo y edad
        mediana_estatal = features.groupby(["entidad_cve", "sexo", "grupo_edad"])[col].transform("median")
        features[col] = features[col].fillna(mediana_estatal)
        # Si aún hay nulos (entidad sin datos), mediana global del grupo de edad y sexo
        mediana_global = features.groupby(["sexo", "grupo_edad"])[col].transform("median")
        features[col] = features[col].fillna(mediana_global)

    # Métricas síntesis
    features["part_media"] = features[cols_part].mean(axis=1)
    features["part_volatilidad"] = features[cols_part].std(axis=1).fillna(0.0)
    features["part_presidencial"] = features[cols_pres].mean(axis=1)
    features["part_intermedia"] = features[cols_inter].mean(axis=1)
    features["brecha_pres_inter"] = features["part_presidencial"] - features["part_intermedia"]
    features["tendencia_2009_2024"] = features["part_2024"] - features["part_2009"]

    # Peso demográfico: lista nominal promedio
    features["ln_promedio"] = features[cols_ln].mean(axis=1).fillna(0).round().astype("int64")

    # Cruzar con nombres de municipio y estado
    features = features.merge(nombres, on=["entidad_cve", "municipio_cve"], how="left")
    features["sexo_desc"] = features["sexo"].map(SEXO_MAP).astype("string")

    # Ordenar columnas limpias
    cols_orden = [
        "entidad_cve",
        "entidad",
        "municipio_cve",
        "municipio",
        "sexo",
        "sexo_desc",
        "grupo_edad",
        "ln_promedio",
        "part_media",
        "part_volatilidad",
        "part_presidencial",
        "part_intermedia",
        "brecha_pres_inter",
        "tendencia_2009_2024",
    ] + cols_part

    features = features[cols_orden]

    # Guardar
    salida_parquet = CLEAN / "matriz_features.parquet"
    features.to_parquet(salida_parquet, index=False)
    print(f"Matriz de features guardada con éxito en {salida_parquet}")
    print(f"Total de registros (unidades demográfico-territoriales): {len(features):,}")
    print("\nResumen descriptivo de las variables clave:")
    cols_resumen = [
        "part_media",
        "part_volatilidad",
        "part_presidencial",
        "part_intermedia",
        "brecha_pres_inter",
        "tendencia_2009_2024",
    ]
    print(features[cols_resumen].describe().round(4).to_string())

    # Exportar resumen para reporte de calidad
    features[cols_resumen].describe().round(4).to_csv(CALIDAD / "resumen_features.csv")
    return features


if __name__ == "__main__":
    construir_features()
