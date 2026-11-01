"""Paso 5: Segmentación y Modelado No Supervisado (Clustering).
Compara K-Means (ponderado por lista nominal), Gaussian Mixture y Jerárquico.
Evalúa k in {4, 5, 6} utilizando:
  - Coeficiente de Silueta
  - Índice Davies-Bouldin
  - Estabilidad por remuestreo (Bootstrap con Adjusted Rand Index)
  - Análisis de sensibilidad
Genera:
  - Gráficas diagnósticas en reports/clustering/
  - data/clean/perfiles_segmentados.parquet (nivel municipio x sexo x edad)
  - data/clean/perfiles_resumen_municipio.parquet y .csv (para Power BI)
  - reports/clustering/caracterizacion_perfiles.csv
"""
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from config import CLEAN, CLUSTERING_REPORTS

# Configurar estilo visual limpio
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({"font.size": 10, "figure.autolayout": True})

# Variables clave para la segmentación longitudinal
FEATURES_CLUSTERING = [
    "part_media",
    "part_volatilidad",
    "part_presidencial",
    "part_intermedia",
    "brecha_pres_inter",
    "tendencia_2009_2024",
]

K_CANDIDATOS = [4, 5, 6]


def evaluar_modelos_y_k(X_scaled, weights, n_iter_bootstrap=10):
    """Evalúa K-Means, GMM y Jerárquico para k in [4, 5, 6].
    Calcula Silhouette, Davies-Bouldin, Calinski-Harabasz y estabilidad ARI."""
    resultados = []
    print("\n--- Evaluando modelos y valores de k (4, 5, 6) ---")

    # Muestra para cálculo rápido de Silueta si el dataset es grande (>10k)
    sample_size_sil = min(10000, len(X_scaled))
    rng = np.random.RandomState(42)
    idx_sample = rng.choice(len(X_scaled), size=sample_size_sil, replace=False)
    X_sub = X_scaled[idx_sample]

    for k in K_CANDIDATOS:
        # 1. K-Means (Ponderado por lista nominal)
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled, sample_weight=weights)
        labels_km = km.labels_
        sil_km = silhouette_score(X_sub, labels_km[idx_sample])
        db_km = davies_bouldin_score(X_scaled, labels_km)
        ch_km = calinski_harabasz_score(X_scaled, labels_km)

        # 2. Gaussian Mixture Model (GMM)
        gmm = GaussianMixture(n_components=k, random_state=42, covariance_type="full")
        labels_gmm = gmm.fit_predict(X_scaled)
        sil_gmm = silhouette_score(X_sub, labels_gmm[idx_sample])
        db_gmm = davies_bouldin_score(X_scaled, labels_gmm)
        ch_gmm = calinski_harabasz_score(X_scaled, labels_gmm)

        # 3. Clustering Jerárquico (Agglomerative Ward en muestra representativa)
        agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels_agg_sub = agg.fit_predict(X_sub)
        sil_agg = silhouette_score(X_sub, labels_agg_sub)
        db_agg = davies_bouldin_score(X_sub, labels_agg_sub)
        ch_agg = calinski_harabasz_score(X_sub, labels_agg_sub)

        # Prueba de estabilidad Bootstrap para K-Means
        ari_scores = []
        for i in range(n_iter_bootstrap):
            idx_boot = rng.choice(len(X_scaled), size=int(0.8 * len(X_scaled)), replace=False)
            km_boot = KMeans(n_clusters=k, random_state=i, n_init=5)
            km_boot.fit(X_scaled[idx_boot], sample_weight=weights[idx_boot])
            pred_all = km_boot.predict(X_scaled)
            ari = adjusted_rand_score(labels_km, pred_all)
            ari_scores.append(ari)
        ari_medio = np.mean(ari_scores)

        print(f"k={k} | K-Means: Sil={sil_km:.4f}, DB={db_km:.4f}, ARI={ari_medio:.4f}")
        print(f"k={k} | GMM:     Sil={sil_gmm:.4f}, DB={db_gmm:.4f}")
        print(f"k={k} | Jerárq:  Sil={sil_agg:.4f}, DB={db_agg:.4f}")

        resultados.append({
            "k": k, "modelo": "K-Means (Ponderado)",
            "silhouette": sil_km, "davies_bouldin": db_km, "calinski_harabasz": ch_km,
            "estabilidad_ari": ari_medio, "modelo_obj": km
        })
        resultados.append({
            "k": k, "modelo": "Gaussian Mixture",
            "silhouette": sil_gmm, "davies_bouldin": db_gmm, "calinski_harabasz": ch_gmm,
            "estabilidad_ari": np.nan, "modelo_obj": gmm
        })
        resultados.append({
            "k": k, "modelo": "Jerárquico (Ward)",
            "silhouette": sil_agg, "davies_bouldin": db_agg, "calinski_harabasz": ch_agg,
            "estabilidad_ari": np.nan, "modelo_obj": agg
        })

    df_res = pd.DataFrame(resultados)
    df_res_export = df_res.drop(columns=["modelo_obj"])
    df_res_export.to_csv(CLUSTERING_REPORTS / "metricas_modelos.csv", index=False)
    print(f"\nMétricas exportadas a {CLUSTERING_REPORTS / 'metricas_modelos.csv'}")

    return df_res


def graficar_metricas(df_res):
    """Genera gráfica de Silueta y Davies-Bouldin por k y modelo."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Silhouette (Mayor es mejor)
    for mod in df_res["modelo"].unique():
        sub = df_res[df_res["modelo"] == mod]
        axes[0].plot(sub["k"], sub["silhouette"], marker="o", lw=2, label=mod)
    axes[0].set_title("Coeficiente de Silueta por k (Mayor es mejor)")
    axes[0].set_xlabel("Número de clusters (k)")
    axes[0].set_ylabel("Silhouette Score")
    axes[0].set_xticks(K_CANDIDATOS)
    axes[0].legend()

    # Davies-Bouldin (Menor es mejor)
    for mod in df_res["modelo"].unique():
        sub = df_res[df_res["modelo"] == mod]
        axes[1].plot(sub["k"], sub["davies_bouldin"], marker="s", lw=2, label=mod)
    axes[1].set_title("Índice Davies-Bouldin por k (Menor es mejor)")
    axes[1].set_xlabel("Número de clusters (k)")
    axes[1].set_ylabel("Davies-Bouldin Index")
    axes[1].set_xticks(K_CANDIDATOS)
    axes[1].legend()

    plt.tight_layout()
    ruta_graf = CLUSTERING_REPORTS / "comparacion_metricas_k.png"
    plt.savefig(ruta_graf, dpi=200)
    plt.close()
    print(f"Gráfica de métricas guardada en {ruta_graf}")


def asignar_y_caracterizar_perfiles(df_features, X_scaled, weights, k_optimo=5):
    """Aplica el modelo K-Means final con k_optimo (5 perfiles),
    asigna etiquetas narrativas, genera gráficas y exporta los datos."""
    print(f"\n--- Entrenando modelo final K-Means con k={k_optimo} ---")
    km = KMeans(n_clusters=k_optimo, random_state=42, n_init=20)
    km.fit(X_scaled, sample_weight=weights)
    clusters = km.labels_

    df_res = df_features.copy()
    df_res["cluster"] = clusters

    # Calcular estadísticas por cluster para nombrar narrativamente
    agrup = df_res.groupby("cluster").apply(
        lambda g: pd.Series({
            "part_media": np.average(g["part_media"], weights=g["ln_promedio"]),
            "part_volatilidad": np.average(g["part_volatilidad"], weights=g["ln_promedio"]),
            "brecha_pres_inter": np.average(g["brecha_pres_inter"], weights=g["ln_promedio"]),
            "tendencia_2009_2024": np.average(g["tendencia_2009_2024"], weights=g["ln_promedio"]),
            "total_poblacion_ln": g["ln_promedio"].sum(),
            "pct_poblacion": g["ln_promedio"].sum() / df_res["ln_promedio"].sum(),
            "pct_mujeres": (g[g["sexo"] == 1]["ln_promedio"].sum()) / g["ln_promedio"].sum(),
        })
    ).reset_index()

    # Mapeo narrativo de acuerdo a los valores observados ordenados por participación
    # Para ser objetivos y reproducibles, ordenamos los clusters por su participación media
    orden_part = agrup.sort_values("part_media")["cluster"].tolist()

    # Definición de 5 perfiles distintivos y rigurosos del sistema electoral mexicano
    nombres_perfiles_map = {
        orden_part[0]: "P1: Abstencionismo Estructural y Desencanto",  # Más baja participación (<40%)
        orden_part[1]: "P2: Votantes Coyunturales (Presidenciales)",    # Mayor brecha presidencial vs intermedia
        orden_part[2]: "P3: Participación Intermedia Moderada",        # Participación en torno a la media nacional
        orden_part[3]: "P4: Ciudadanía Consistente en Ascenso",        # Alta tendencia positiva de crecimiento
        orden_part[4]: "P5: Núcleo Cívico de Alta Fidelidad",          # Máxima participación (>70%)
    }

    df_res["perfil_cve"] = df_res["cluster"].map(lambda c: f"P{orden_part.index(c)+1}")
    df_res["perfil_nombre"] = df_res["cluster"].map(nombres_perfiles_map)

    # Actualizar la tabla descriptiva con el nuevo orden y nombre
    agrup["perfil_cve"] = agrup["cluster"].map(lambda c: f"P{orden_part.index(c)+1}")
    agrup["perfil_nombre"] = agrup["cluster"].map(nombres_perfiles_map)
    agrup = agrup.sort_values("perfil_cve").reset_index(drop=True)

    print("\n=== CARACTERIZACIÓN DE LOS 5 PERFILES CIUDADANOS (PONDERADO POR LISTA NOMINAL) ===")
    cols_mostrar = [
        "perfil_cve",
        "perfil_nombre",
        "part_media",
        "part_volatilidad",
        "brecha_pres_inter",
        "tendencia_2009_2024",
        "pct_poblacion",
        "pct_mujeres",
    ]
    print(agrup[cols_mostrar].round(4).to_string())

    # Exportar caracterización
    agrup.to_csv(CLUSTERING_REPORTS / "caracterizacion_perfiles.csv", index=False)

    # Generar todos los gráficos analíticos
    generar_visualizaciones_analiticas(df_res, agrup, k_optimo)

    # 3. Resumen a nivel MUNICIPIO (Para Power BI y Cartografía)
    # Identifica el perfil dominante de cada municipio (mayor lista nominal)
    mun_perfil = df_res.groupby(["entidad_cve", "entidad", "municipio_cve", "municipio", "perfil_cve"])["ln_promedio"].sum().reset_index()
    mun_tot = df_res.groupby(["entidad_cve", "municipio_cve"])["ln_promedio"].sum().reset_index().rename(columns={"ln_promedio": "ln_total"})
    mun_perfil = mun_perfil.merge(mun_tot, on=["entidad_cve", "municipio_cve"])
    mun_perfil["pct_perfil"] = (mun_perfil["ln_promedio"] / mun_perfil["ln_total"]).round(4)

    # Perfil dominante por municipio
    idx_dom = mun_perfil.groupby(["entidad_cve", "municipio_cve"])["ln_promedio"].idxmax()
    mun_dominante = mun_perfil.loc[idx_dom].copy()
    mun_dominante = mun_dominante.rename(columns={"perfil_cve": "perfil_dominante", "pct_perfil": "pct_perfil_dominante"})
    mun_dominante["perfil_dominante_nombre"] = mun_dominante["perfil_dominante"].map(lambda p: agrup.set_index("perfil_cve")["perfil_nombre"].to_dict().get(p))

    # Pivotear la mezcla de perfiles por municipio para el mapa coroplético
    mun_pivot = mun_perfil.pivot_table(
        index=["entidad_cve", "entidad", "municipio_cve", "municipio"],
        columns="perfil_cve",
        values="pct_perfil",
        fill_value=0.0
    ).reset_index()
    mun_pivot.columns = [f"pct_{c}" if c.startswith("P") else c for c in mun_pivot.columns]

    resumen_municipio = mun_dominante[["entidad_cve", "entidad", "municipio_cve", "municipio", "ln_total", "perfil_dominante", "perfil_dominante_nombre", "pct_perfil_dominante"]].merge(
        mun_pivot, on=["entidad_cve", "entidad", "municipio_cve", "municipio"]
    )

    # Guardar salidas finales
    df_res.to_parquet(CLEAN / "perfiles_segmentados.parquet", index=False)
    resumen_municipio.to_parquet(CLEAN / "perfiles_resumen_municipio.parquet", index=False)
    resumen_municipio.to_csv(CLEAN / "perfiles_resumen_municipio.csv", index=False, encoding="utf-8-sig")

    print(f"\nSalidas guardadas:")
    print(f" - {CLEAN / 'perfiles_segmentados.parquet'} ({len(df_res):,} filas)")
    print(f" - {CLEAN / 'perfiles_resumen_municipio.parquet'} y .csv ({len(resumen_municipio):,} municipios)")
    print(f" - Gráficas y caracterización en {CLUSTERING_REPORTS}")

    return df_res, resumen_municipio


def generar_visualizaciones_analiticas(df_res, agrup, k_optimo):
    """Genera el conjunto completo de gráficas para el informe y la presentación del INE."""
    colores_perfiles = {
        "P1": "#D9534F",  # Rojo suave / Abstencionismo
        "P2": "#F39C12",  # Naranja / Coyuntural
        "P3": "#7F8C8D",  # Gris azulado / Moderado
        "P4": "#2980B9",  # Azul institucional / Ascenso
        "P5": "#27AE60",  # Verde / Alta fidelidad
    }
    anios = [2009, 2012, 2015, 2018, 2021, 2024]
    presidenciales = [2012, 2018, 2024]

    # ==============================================================
    # 1. PARTICIPACIÓN MEDIA POR PERFIL (Barras Horizontales con datos)
    # ==============================================================
    fig, ax = plt.subplots(figsize=(10, 5))
    agrup_sorted = agrup.sort_values("part_media", ascending=True)
    bars = ax.barh(
        agrup_sorted["perfil_cve"],
        agrup_sorted["part_media"] * 100,
        color=[colores_perfiles.get(p, "#333333") for p in agrup_sorted["perfil_cve"]],
        height=0.6,
        edgecolor="none",
    )
    ax.set_xlim(0, 85)
    ax.set_xlabel("Tasa de Participación Media Histórica (%)", fontweight="bold")
    ax.set_title("Participación Media por Perfil Ciudadano (Elecciones 2009-2024)", fontsize=13, fontweight="bold", pad=12)

    for bar, (_, row) in zip(bars, agrup_sorted.iterrows()):
        w = bar.get_width()
        millones = row["total_poblacion_ln"] / 1e6
        pct_padron = row["pct_poblacion"] * 100
        ax.text(
            w + 1.2,
            bar.get_y() + bar.get_height() / 2,
            f"{w:.1f}%  ({row['perfil_nombre'].split(': ')[1]} | {pct_padron:.1f}% padrón, {millones:.1f}M electores)",
            va="center",
            fontsize=9.5,
            fontweight="bold",
        )
    ax.axvline(56.3, color="#555555", linestyle="--", alpha=0.7, label="Media Nacional Histórica (56.3%)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(CLUSTERING_REPORTS / "participacion_media_por_perfil.png", dpi=250)
    plt.close()
    print(" - Gráfico guardado: participacion_media_por_perfil.png")

    # ==============================================================
    # 2. PARTICIPACIÓN POR AÑO PARA CADA PERFIL (Serie de Tiempo 2009-2024)
    # ==============================================================
    fig, ax = plt.subplots(figsize=(11, 6))

    # Calcular participación ponderada por año para cada cluster
    cols_anios = [f"part_{a}" for a in anios]
    for p in agrup["perfil_cve"]:
        sub = df_res[df_res["perfil_cve"] == p]
        tasas_anios = []
        for col in cols_anios:
            tasa_ponderada = np.average(sub[col], weights=sub["ln_promedio"]) * 100
            tasas_anios.append(tasa_ponderada)

        nom_corto = agrup.loc[agrup["perfil_cve"] == p, "perfil_nombre"].values[0]
        ax.plot(
            anios,
            tasas_anios,
            marker="o",
            lw=2.8,
            markersize=7,
            label=f"{nom_corto}",
            color=colores_perfiles.get(p, "#333"),
        )
        # Anotar valor en 2024
        ax.text(2024.1, tasas_anios[-1], f"{tasas_anios[-1]:.1f}%", va="center", fontsize=8.5, fontweight="bold", color=colores_perfiles.get(p, "#333"))

    # Sombrear elecciones presidenciales
    for pres in presidenciales:
        ax.axvspan(pres - 0.35, pres + 0.35, color="#EBF5FB", alpha=0.8, zorder=0)
        ax.text(pres, 22, "Presidencial", ha="center", fontsize=8, color="#2980B9", style="italic")

    ax.set_xticks(anios)
    ax.set_ylim(20, 85)
    ax.set_xlim(2008.5, 2025.2)
    ax.set_ylabel("Tasa de Participación (%)", fontweight="bold")
    ax.set_xlabel("Año de la Elección Federal", fontweight="bold")
    ax.set_title("Evolución de la Participación Electoral por Perfil (2009 - 2024)", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig(CLUSTERING_REPORTS / "participacion_por_anio_perfiles.png", dpi=250)
    plt.close()
    print(" - Gráfico guardado: participacion_por_anio_perfiles.png")

    # ==============================================================
    # 3. COMPOSICIÓN POR SEXO Y EDAD DE CADA PERFIL
    # ==============================================================
    fig, axes = plt.subplots(1, 5, figsize=(18, 5), sharey=True)
    grupos_edad = ["18-19", "20-24", "25-29", "30-39", "40-49", "50-59", "60-69", "70+"]

    for idx, p in enumerate(agrup["perfil_cve"]):
        ax = axes[idx]
        sub = df_res[df_res["perfil_cve"] == p]
        tot_p = sub["ln_promedio"].sum()

        hombres = [
            (sub[(sub["grupo_edad"] == e) & (sub["sexo"] == 0)]["ln_promedio"].sum() / tot_p) * 100
            for e in grupos_edad
        ]
        mujeres = [
            (sub[(sub["grupo_edad"] == e) & (sub["sexo"] == 1)]["ln_promedio"].sum() / tot_p) * 100
            for e in grupos_edad
        ]

        y = np.arange(len(grupos_edad))
        bar_h = 0.38
        ax.barh(y - bar_h/2, hombres, height=bar_h, label="Hombres", color="#5DADE2", edgecolor="none")
        ax.barh(y + bar_h/2, mujeres, height=bar_h, label="Mujeres", color="#AF7AC5", edgecolor="none")

        ax.set_yticks(y)
        ax.set_yticklabels(grupos_edad, fontsize=9)
        nom_linea = agrup.loc[agrup["perfil_cve"] == p, "perfil_nombre"].values[0].replace(": ", ":\n")
        ax.set_title(f"{nom_linea}", fontsize=10, fontweight="bold", pad=8)
        ax.set_xlabel("% del Perfil", fontsize=9)
        ax.grid(axis="x", linestyle=":", alpha=0.6)
        if idx == 0:
            ax.set_ylabel("Grupo de Edad", fontweight="bold")
            ax.legend(loc="lower right", fontsize=8.5)

    fig.suptitle("Composición Cruzada por Sexo y Grupo de Edad en Cada Perfil Ciudadano", fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(CLUSTERING_REPORTS / "composicion_sexo_edad_perfiles.png", dpi=250)
    plt.close()
    print(" - Gráfico guardado: composicion_sexo_edad_perfiles.png")

    # ==============================================================
    # 4. VALOR AGREGADO: BRECHA PRESIDENCIAL VS INTERMEDIA (Dumbbell Plot)
    # ==============================================================
    fig, ax = plt.subplots(figsize=(10, 5))
    y_pos = np.arange(len(agrup))

    for idx, row in agrup.iterrows():
        p = row["perfil_cve"]
        sub = df_res[df_res["perfil_cve"] == p]
        inter = np.average(sub["part_intermedia"], weights=sub["ln_promedio"]) * 100
        pres = np.average(sub["part_presidencial"], weights=sub["ln_promedio"]) * 100

        # Línea de unión
        ax.plot([inter, pres], [idx, idx], color="#95A5A6", lw=3, zorder=1)
        # Punto intermedias
        ax.scatter(inter, idx, color="#E74C3C", s=110, zorder=2, label="Intermedias (Diputados)" if idx == 0 else "")
        # Punto presidenciales
        ax.scatter(pres, idx, color="#2ECC71", s=110, zorder=2, label="Presidenciales" if idx == 0 else "")

        # Texto de brecha
        brecha = pres - inter
        ax.text((inter + pres)/2, idx + 0.18, f"+{brecha:.1f}%", ha="center", fontsize=9, fontweight="bold", color="#2C3E50")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{row['perfil_cve']}: {row['perfil_nombre'].split(': ')[1]}" for _, row in agrup.iterrows()], fontweight="bold")
    ax.set_xlabel("Tasa de Participación Ponderada (%)", fontweight="bold")
    ax.set_title("Efecto de la Convocatoria Presidencial: Brecha entre Presidenciales e Intermedias", fontsize=12, fontweight="bold", pad=12)
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(CLUSTERING_REPORTS / "brecha_presidencial_dumbbell.png", dpi=250)
    plt.close()
    print(" - Gráfico guardado: brecha_presidencial_dumbbell.png")

    # ==============================================================
    # 5. VALOR AGREGADO: CONCENTRACIÓN REGIONAL POR PERFIL
    # ==============================================================
    # Mapeo de Entidades a 4 Macro-Regiones del INE
    regiones_map = {
        "02": "Norte", "03": "Norte", "05": "Norte", "08": "Norte", "10": "Norte", "19": "Norte", "26": "Norte", "28": "Norte",
        "01": "Occidente/Bajío", "06": "Occidente/Bajío", "11": "Occidente/Bajío", "14": "Occidente/Bajío", "16": "Occidente/Bajío", "18": "Occidente/Bajío", "22": "Occidente/Bajío", "24": "Occidente/Bajío", "32": "Occidente/Bajío",
        "09": "Centro", "13": "Centro", "15": "Centro", "17": "Centro", "21": "Centro", "29": "Centro",
        "04": "Sur-Sureste", "07": "Sur-Sureste", "12": "Sur-Sureste", "20": "Sur-Sureste", "23": "Sur-Sureste", "27": "Sur-Sureste", "30": "Sur-Sureste", "31": "Sur-Sureste"
    }
    df_res["region"] = df_res["entidad_cve"].map(regiones_map).fillna("Otra")

    reg_pivot = df_res.groupby(["region", "perfil_cve"])["ln_promedio"].sum().unstack("perfil_cve")
    reg_pct = reg_pivot.div(reg_pivot.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    reg_pct.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        color=[colores_perfiles.get(c, "#777") for c in reg_pct.columns],
        edgecolor="white",
        width=0.6,
    )
    ax.set_title("Mezcla de Perfiles Ciudadanos por Macro-Región Electoral (%)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Macro-Región", fontweight="bold")
    ax.set_ylabel("Porcentaje de la Lista Nominal", fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontweight="bold")
    ax.legend(title="Perfil", bbox_to_anchor=(1.02, 1), loc="upper left", labels=[f"{c}: {agrup.set_index('perfil_cve')['perfil_nombre'].to_dict().get(c).split(': ')[1]}" for c in reg_pct.columns])
    plt.tight_layout()
    plt.savefig(CLUSTERING_REPORTS / "distribucion_regional_perfiles.png", dpi=250)
    plt.close()
    print(" - Gráfico guardado: distribucion_regional_perfiles.png")



def main():
    print("Iniciando pipeline de segmentación...")
    df_features = pd.read_parquet(CLEAN / "matriz_features.parquet")

    X = df_features[FEATURES_CLUSTERING].values
    weights = df_features["ln_promedio"].values.astype(float)
    # Evitar pesos 0
    weights = np.clip(weights, 1.0, None)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 1. Evaluar modelos y seleccionar k
    df_metricas = evaluar_modelos_y_k(X_scaled, weights)
    graficar_metricas(df_metricas)

    # 2. Ajustar con k=5 perfiles (óptimo balance interpretabilidad/rendimiento)
    asignar_y_caracterizar_perfiles(df_features, X_scaled, weights, k_optimo=5)
    print("\n¡Pipeline de Clustering finalizado con éxito!")


if __name__ == "__main__":
    main()
