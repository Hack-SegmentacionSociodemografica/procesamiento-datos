"""Paso 1b. Consideraciones de modalidad de voto y prisión preventiva.
Lee los CSV por estado (desde zip o sueltos) de data/raw/voto_anticipado/ y data/raw/prision_preventiva/,
los limpia y los agrega. Cada fila cuenta como UN registro"""
import zipfile
from pathlib import Path

import pandas as pd

from config import CALIDAD, CLEAN, RAW

COL_CONTEO = None

MODALIDADES = {
    "voto_anticipado": {
        "claves": ["anio", "entidad_cve", "municipio_cve", "sexo"],
        "salida": "voto_anticipado_municipio.parquet", "total": "votos_anticipados"},
    "voto_prisión_preventiva": {
        "claves": ["anio", "entidad_cve", "distrito_federal", "sexo"],
        "salida": "prision_preventiva_distrito.parquet", "total": "votos_prision_preventiva"},
}
RENAME = {"AELEC": "anio", "EDOCVE": "entidad_cve", "ESTADO": "entidad_cve", "EDONOM": "entidad", "MPIOCVE": "municipio_cve",
          "MPIONOM": "municipio", "DEF": "distrito_federal", "DEL": "distrito_local", "SEXO": "sexo"}
PEDIDAS = set(RENAME) | ({COL_CONTEO.strip().upper()} if COL_CONTEO else set())


def norm(c):
    return str(c).replace("\ufeff", "").strip().upper()


def fuentes(carpeta: Path):
    zips = sorted(carpeta.glob("*.zip"))
    if zips:
        for z in zips:
            with zipfile.ZipFile(z) as zf:
                nombres = [n for n in zf.namelist() if n.lower().endswith(".csv") and "__MACOSX" not in n]
            for n in sorted(nombres):
                yield Path(n).name, (lambda z=z, n=n: zipfile.ZipFile(z).open(n))
    else:
        for f in sorted(carpeta.rglob("*.csv")):
            yield f.name, (lambda f=f: open(f, "rb"))


def leer(opener):
    with opener() as fh:
        linea = fh.readline().decode("latin-1")
    sep = max([",", ";", "|", "\t"], key=linea.count)
    for enc in ("utf-8-sig", "latin-1"):
        try:
            with opener() as fh:
                return pd.read_csv(fh, sep=sep, dtype=str, encoding=enc, low_memory=False,
                                   usecols=lambda c: norm(c) in PEDIDAS)
        except UnicodeDecodeError:
            continue
    raise ValueError("No se pudo leer el archivo")


def procesar(nombre, opener, cfg):
    df = leer(opener)
    df.columns = [("_conteo" if COL_CONTEO and norm(c) == COL_CONTEO.strip().upper() else RENAME[norm(c)])
                  for c in df.columns]
    faltan = [c for c in cfg["claves"] if c not in df.columns]
    if faltan:
        raise SystemExit(f"{nombre}: faltan columnas {faltan}. Encontradas: {list(df.columns)}")
    for c in df.columns:
        df[c] = df[c].astype("string").str.strip()
    n0 = len(df)

    df["anio"] = pd.to_numeric(df["anio"], errors="coerce").astype("Int16")
    df["sexo"] = pd.to_numeric(df["sexo"], errors="coerce").astype("Int8")
    df["entidad_cve"] = df["entidad_cve"].str.replace(r"\.0$", "", regex=True).str.zfill(2)
    if "municipio_cve" in df.columns:
        df["municipio_cve"] = df["municipio_cve"].str.replace(r"\.0$", "", regex=True).str.zfill(3)
    if "distrito_federal" in df.columns:
        df["distrito_federal"] = df["distrito_federal"].str.replace(r"\.0$", "", regex=True)

    df["_n"] = pd.to_numeric(df["_conteo"], errors="coerce") if COL_CONTEO else 1
    invalida = df["anio"].isna() | ~df["sexo"].isin([0, 1, 2]) | pd.Series(df["_n"]).isna() \
        | (pd.Series(df["_n"]) < 0).fillna(False)
    stats = {"archivo": nombre, "filas_leidas": n0, "filas_descartadas": int(invalida.sum())}

    df = df[~invalida]
    agg = df.groupby(cfg["claves"], dropna=False, sort=False)["_n"].sum().rename(cfg["total"]).reset_index()
    return agg, stats


def main():
    resumen, log = [], []
    for nombre_mod, cfg in MODALIDADES.items():
        carpeta = RAW / nombre_mod
        if not carpeta.exists():
            print(f"[{nombre_mod}] no existe {carpeta}, se omite")
            continue
        partes = []
        for nombre, opener in fuentes(carpeta):
            agg, st = procesar(nombre, opener, cfg)
            partes.append(agg); log.append({"modalidad": nombre_mod, **st})
            print(f"[{nombre_mod}] {nombre}: {st['filas_leidas']:,} filas (descartadas: {st['filas_descartadas']:,})")
        if not partes:
            print(f"[{nombre_mod}] sin archivos csv")
            continue
        base = pd.concat(partes, ignore_index=True)
        base = base.groupby(cfg["claves"], dropna=False, sort=False)[cfg["total"]].sum().reset_index()
        base.to_parquet(CLEAN / cfg["salida"], index=False)

        r = base.groupby(["anio", "sexo"])[cfg["total"]].sum().reset_index()
        r.insert(0, "modalidad", nombre_mod)
        r = r.rename(columns={cfg["total"]: "registros"})
        resumen.append(r)
        print(f"\n[{nombre_mod}] guardado {cfg['salida']} ({len(base):,} filas). Registros por año y sexo:")
        print(r.to_string(index=False), "\n")

    if log:
        pd.DataFrame(log).to_csv(CALIDAD / "calidad_modalidades_por_archivo.csv", index=False)
    if resumen:
        pd.concat(resumen).to_csv(CALIDAD / "resumen_modalidades.csv", index=False)


if __name__ == "__main__":
    main()