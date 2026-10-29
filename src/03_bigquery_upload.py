""" Sube los parquet de data/clean a BigQuery y verifica el número de filas.
Isma antes de todo instala Google Cloud CLI y correr  gcloud auth application-default login"""

import pyarrow.parquet as pq
from google.cloud import bigquery

from config import BQ_DATASET, CLEAN, GCP_PROJECT

TABLAS = {
    "base_limpia": "base_limpia.parquet",
    "base_municipio": "base_municipio.parquet",
    "catalogo_territorial": "catalogo_territorial.parquet",
}


def main():
    client = bigquery.Client(project=GCP_PROJECT)
    client.create_dataset(bigquery.Dataset(f"{GCP_PROJECT}.{BQ_DATASET}"), exists_ok=True)
    cfg = bigquery.LoadJobConfig(source_format=bigquery.SourceFormat.PARQUET,
                                 write_disposition="WRITE_TRUNCATE")
    for tabla, archivo in TABLAS.items():
        ruta = CLEAN / archivo
        esperado = pq.ParquetFile(ruta).metadata.num_rows
        destino = f"{GCP_PROJECT}.{BQ_DATASET}.{tabla}"
        with open(ruta, "rb") as fh:
            client.load_table_from_file(fh, destino, job_config=cfg).result()
        cargadas = client.get_table(destino).num_rows
        print(f"{tabla}: parquet {esperado:,} filas | BigQuery {cargadas:,} filas")
        assert esperado == cargadas, f"El conteo de {tabla} no coincide"
    print("Carga verificada.")


if __name__ == "__main__":
    main()