import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


# agregar carpetas si aun no existen

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
CLEAN = ROOT / "data" / "clean"
CALIDAD = ROOT / "reports" / "calidad"
CLUSTERING_REPORTS = ROOT / "reports" / "clustering"

ANIOS = [2009, 2012, 2015, 2018, 2021, 2024]
PRESIDENCIALES = [2012, 2018, 2024]
INTERMEDIAS = [2009, 2015, 2021]

GCP_PROJECT = os.getenv("GCP_PROJECT_ID", "tu-proyecto-id")
BQ_DATASET = os.getenv("BQ_DATASET", "participacion")

# Grupos de edad
EDAD_BINS = [18, 20, 25, 30, 40, 50, 60, 70, 200]
EDAD_LABELS = ["18-19", "20-24", "25-29", "30-39", "40-49", "50-59", "60-69", "70+"]

SEXO_MAP = {0: "Hombres", 1: "Mujeres", 2: "No Binario"}

for p in (RAW, INTERIM, CLEAN, CALIDAD, CLUSTERING_REPORTS):
    p.mkdir(parents=True, exist_ok=True)
