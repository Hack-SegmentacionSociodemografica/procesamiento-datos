"""Paso 1. Lee los CSV de los 32 estados de cada año (desde el zip o sueltos) y guarda
un parquet por archivo en data/interim/<año>/. Todo se procesa archivo por archivo."""
import sys
import zipfile
from pathlib import Path

import pandas as pd

from config import ANIOS, CALIDAD, INTERIM, RAW

RENAME = {
    "AELEC": "anio", "FELECCION": "fecha_eleccion", "EDOCVE": "entidad_cve", "EDONOM": "entidad",
    "MPIOCVE": "municipio_cve", "MPIONOM": "municipio", "SECCION": "seccion", "TIPOSEC": "tipo_seccion",
    "DEF": "distrito_federal", "DEL": "distrito_local", "SEXO": "sexo", "EDAD": "edad",
    "LN": "ln", "SV": "sv", "NV": "nv", "NS": "ns",
}
OBLIGATORIAS = ["anio", "entidad_cve", "municipio_cve", "seccion", "sexo", "edad", "ln", "sv", "nv", "ns"]
OPCIONALES = ["entidad", "municipio", "tipo_seccion", "distrito_federal", "distrito_local"]
NUMERICAS = {"sexo": "Int8", "edad": "Int16", "ln": "Int32", "sv": "Int32", "nv": "Int32", "ns": "Int32"}


def norm(c):
    return str(c).replace("\ufeff", "").strip().upper()


def fuentes(carpeta: Path):
    """Evita duplicacion de datos"""
    zips = sorted(carpeta.glob("*.zip"))
    if zips:
        for z in zips:
            with zipfile.ZipFile(z) as zf:
                nombres = [n for n in zf.namelist()
                           if n.lower().endswith(".csv") and "__MACOSX" not in n]
            for n in sorted(nombres):
                yield Path(n).name, (lambda z=z, n=n: zipfile.ZipFile(z).open(n))
    else:
        for f in sorted(carpeta.rglob("*.csv")):
            yield f.name, (lambda f=f: open(f, "rb"))


def detectar_sep(opener):
    with opener() as fh:
        linea = fh.readline().decode("latin-1")
    return max([",", ";", "|", "\t"], key=linea.count)


def leer(opener):
    sep = detectar_sep(opener)
    for enc in ("utf-8-sig", "latin-1"):
        try:
            with opener() as fh:
                return pd.read_csv(fh, sep=sep, dtype=str, encoding=enc, low_memory=False,
                                   usecols=lambda c: norm(c) in RENAME)
        except UnicodeDecodeError:
            continue
    raise ValueError("No se pudo leer el archivo con utf-8 ni latin-1")


def procesar(anio, nombre, opener):
    df = leer(opener)
    df.columns = [RENAME[norm(c)] for c in df.columns]
    faltan = [c for c in OBLIGATORIAS if c not in df.columns]
    if faltan:
        raise SystemExit(f"[{anio}] {nombre}: faltan columnas {faltan}. Encontradas: {list(df.columns)}")
    for c in OPCIONALES:
        if c not in df.columns:
            df[c] = pd.NA
    df = df.drop(columns=[c for c in ("fecha_eleccion",) if c in df.columns])

    for c in df.columns:
        df[c] = df[c].astype("string").str.strip()

    anio_distinto = int((df["anio"] != str(anio)).sum())
    df["anio"] = pd.Series(anio, index=df.index).astype("int16")

    no_num = 0
    for c, t in NUMERICAS.items():
        s = pd.to_numeric(df[c], errors="coerce")
        no_num += int(s.isna().sum() - df[c].isna().sum())
        df[c] = s.astype(t)

    destino = INTERIM / str(anio) / f"{Path(nombre).stem}.parquet"
    destino.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(destino, index=False)
    return {"anio": anio, "archivo": nombre, "filas": len(df),
            "aelec_distinto_de_carpeta": anio_distinto, "valores_no_numericos": no_num}


def main(anios):
    for anio in anios:
        carpeta = RAW / str(anio)
        if not carpeta.exists():
            print(f"[{anio}] no existe {carpeta}, se omite")
            continue
        log = []
        for nombre, opener in fuentes(carpeta):
            r = procesar(anio, nombre, opener)
            log.append(r)
            print(f"[{anio}] {nombre}: {r['filas']:,} filas")
        if log:
            pd.DataFrame(log).to_csv(CALIDAD / f"log_ingesta_{anio}.csv", index=False)
            print(f"[{anio}] listo: {len(log)} archivos, {sum(x['filas'] for x in log):,} filas\n")


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or ANIOS)