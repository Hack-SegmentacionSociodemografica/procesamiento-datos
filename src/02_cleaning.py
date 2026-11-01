"""Limpia, valida y agrega los parquet de data/interim.
Salidas en data/clean/:
  - base_limpia.parquet          -> seccion x sexo x grupo de edad x año
  - base_municipio.parquet       -> municipio x sexo x grupo de edad x año (unidad estable entre años)
  - catalogo_territorial.parquet -> nombres de entidades y municipios
Reportes en reports/calidad/."""
import pandas as pd

from config import CALIDAD, CLEAN, EDAD_BINS, EDAD_LABELS, INTERIM

CLAVES = ["anio", "entidad_cve", "municipio_cve", "seccion", "tipo_seccion",
          "distrito_federal", "distrito_local", "sexo", "grupo_edad"]
CONTEOS = ["ln", "sv", "nv", "ns"]


def limpiar_clave(s, ancho=None):
    s = s.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    return s.str.zfill(ancho) if ancho else s


def falso_si_nulo(s):
    return s.fillna(False).astype(bool)


def procesar(archivo):
    df = pd.read_parquet(archivo)
    n0 = len(df)

    df["entidad_cve"] = limpiar_clave(df["entidad_cve"], 2)
    df["municipio_cve"] = limpiar_clave(df["municipio_cve"], 3)
    df["seccion"] = limpiar_clave(df["seccion"], 4)
    df["tipo_seccion"] = df["tipo_seccion"].astype("string").str.strip().str.upper()
    df["distrito_federal"] = limpiar_clave(df["distrito_federal"])
    df["distrito_local"] = limpiar_clave(df["distrito_local"])

    catalogo = df[["anio", "entidad_cve", "entidad", "municipio_cve", "municipio"]].drop_duplicates()

    nulos_conteo = df[CONTEOS].isna().any(axis=1)
    negativos = falso_si_nulo((df[CONTEOS] < 0).any(axis=1))
    sexo_mal = ~df["sexo"].isin([0, 1, 2])
    edad_mal = df["edad"].isna() | falso_si_nulo(df["edad"] < 18) | falso_si_nulo(df["edad"] >= 200)
    suma_mal = falso_si_nulo(df["ln"] != (df["sv"] + df["nv"] + df["ns"]))
    invalida = nulos_conteo | negativos | sexo_mal | edad_mal

    stats = {"archivo": archivo.name, "anio": int(df["anio"].iloc[0]), "filas_leidas": n0,
             "filas_descartadas": int(invalida.sum()), "conteos_nulos": int(nulos_conteo.sum()),
             "conteos_negativos": int(negativos.sum()), "sexo_invalido": int(sexo_mal.sum()),
             "edad_invalida": int(edad_mal.sum()), "ln_distinto_de_sv_nv_ns": int(suma_mal.sum())}

    df = df[~invalida].copy()
    df["grupo_edad"] = pd.cut(df["edad"].astype("int32"), bins=EDAD_BINS, labels=EDAD_LABELS,
                              right=False).astype("string")
    agg = df.groupby(CLAVES, dropna=False, sort=False)[CONTEOS].sum().reset_index()
    return agg, catalogo, stats


def main():
    archivos = sorted(INTERIM.rglob("*.parquet"))
    if not archivos:
        raise SystemExit("No hay parquet en data/interim. Corre antes 01_ingest.py")

    partes, catalogos, stats = [], [], []
    for a in archivos:
        agg, cat, st = procesar(a)
        partes.append(agg); catalogos.append(cat); stats.append(st)
        print(f"{st['anio']} {a.name}: {st['filas_leidas']:,} -> {len(agg):,} filas agregadas "
              f"(descartadas: {st['filas_descartadas']:,})")

    base = pd.concat(partes, ignore_index=True)
    duplicadas = int(base.duplicated(CLAVES).sum())
    if duplicadas:
        print(f"[AVISO] {duplicadas:,} claves repetidas entre archivos; se suman")
        base = base.groupby(CLAVES, dropna=False, sort=False)[CONTEOS].sum().reset_index()

    mun = (base.groupby(["anio", "entidad_cve", "municipio_cve", "sexo", "grupo_edad"], dropna=False)[CONTEOS]
           .sum().reset_index())
    catalogo = (pd.concat(catalogos, ignore_index=True)
                .drop_duplicates(subset=["anio", "entidad_cve", "municipio_cve"]))

   
    pd.DataFrame(stats).to_csv(CALIDAD / "calidad_por_archivo.csv", index=False)
    nac = base.groupby("anio")[CONTEOS].sum()
    nac["participacion_sv_sobre_ln"] = (nac["sv"] / nac["ln"]).astype("float64").round(4)
    nac["participacion_sv_sobre_sv_nv"] = (nac["sv"] / (nac["sv"] + nac["nv"])).astype("float64").round(4)
    nac.to_csv(CALIDAD / "resumen_nacional.csv")
    (mun.groupby(["entidad_cve", "anio"])["municipio_cve"].nunique().unstack("anio")
        .to_csv(CALIDAD / "municipios_por_entidad_y_anio.csv"))

    base.to_parquet(CLEAN / "base_limpia.parquet", index=False)
    mun.to_parquet(CLEAN / "base_municipio.parquet", index=False)
    catalogo.to_parquet(CLEAN / "catalogo_territorial.parquet", index=False)

    print("\nResumen nacional:")
    print(nac.to_string())
    print(f"\nbase_limpia: {len(base):,} filas | base_municipio: {len(mun):,} filas")


if __name__ == "__main__":
    main()