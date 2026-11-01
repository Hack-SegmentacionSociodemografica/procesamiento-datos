"""Sube los parquet de data/clean a BigQuery y verifica el número de filas.
Requiere gcloud auth o GOOGLE_APPLICATION_CREDENTIALS configurado."""
from pathlib import Path
import pyarrow.parquet as pq
from google.cloud import bigquery

from config import BQ_DATASET, CLEAN, GCP_PROJECT

TABLAS = {
    "base_limpia": "base_limpia.parquet",
    "base_municipio": "base_municipio.parquet",
    "catalogo_territorial": "catalogo_territorial.parquet",
    "voto_anticipado_municipio": "voto_anticipado_municipio.parquet",
    "prision_preventiva_distrito": "prision_preventiva_distrito.parquet",
    "matriz_features": "matriz_features.parquet",
    "perfiles_segmentados": "perfiles_segmentados.parquet",
    "perfiles_resumen_municipio": "perfiles_resumen_municipio.parquet",
}


def main():
    client = bigquery.Client(project=GCP_PROJECT)
    client.create_dataset(bigquery.Dataset(f"{GCP_PROJECT}.{BQ_DATASET}"), exists_ok=True)
    cfg = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.PARQUET,
        write_disposition="WRITE_TRUNCATE",
    )
    for tabla, archivo in TABLAS.items():
        ruta = CLEAN / archivo
        if not ruta.exists():
            print(f"[OMITIDO] {archivo} no existe en data/clean, se continúa.")
            continue
        esperado = pq.ParquetFile(ruta).metadata.num_rows
        destino = f"{GCP_PROJECT}.{BQ_DATASET}.{tabla}"
        with open(ruta, "rb") as fh:
            client.load_table_from_file(fh, destino, job_config=cfg).result()
        cargadas = client.get_table(destino).num_rows
        print(f"{tabla}: parquet {esperado:,} filas | BigQuery {cargadas:,} filas")
        assert esperado == cargadas, f"El conteo de {tabla} no coincide"
    print("Carga y verificación en BigQuery completada.")


if __name__ == "__main__":
    main()
