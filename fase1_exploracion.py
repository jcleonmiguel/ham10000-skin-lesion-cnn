"""
Phase 1 - Dataset exploration: parses the annotations and builds the master CSV (outputs/dataset_info.csv).

TFM - Clasificación de Lesiones Cutáneas (HAM10000)
====================================================
FASE 1: Exploración del dataset y generación del CSV maestro

Estructura esperada:
    TFM CANCER PIEL/
        ds/
            ann/   <- archivos .jpg.json
            img/   <- archivos .jpg
        meta.json

Ejecutar desde la carpeta raíz del proyecto:
    python fase1_exploracion.py
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURACIÓN — ajusta BASE_DIR si es necesario
# ─────────────────────────────────────────────
BASE_DIR  = Path(".")          # carpeta donde está el script (TFM CANCER PIEL)
ANN_DIR   = BASE_DIR / "ds" / "ann"
IMG_DIR   = BASE_DIR / "ds" / "img"
META_FILE = BASE_DIR / "meta.json"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Mapeo de nombre largo → código corto (estándar HAM10000)
CLASS_CODE = {
    "melanocytic nevi"             : "nv",
    "melanoma"                     : "mel",
    "benign keratosis-like lesions": "bkl",
    "basal cell carcinoma"         : "bcc",
    "actinic keratoses"            : "akiec",
    "vascular lesions"             : "vasc",
    "dermatofibroma"               : "df",
}

# ─────────────────────────────────────────────
# 1. LEER META.JSON
# ─────────────────────────────────────────────
print("=" * 60)
print("  TFM - HAM10000  |  FASE 1: Exploración del dataset")
print("=" * 60)

if META_FILE.exists():
    with open(META_FILE, "r") as f:
        meta = json.load(f)
    print(f"\n[meta.json] Clases definidas en el proyecto:")
    for cls in meta.get("classes", []):
        print(f"   • {cls['title']}  (geometría: {cls.get('shape', '?')})")
else:
    print("[AVISO] meta.json no encontrado — continuando sin él.")

# ─────────────────────────────────────────────
# 2. PARSEAR TODOS LOS JSONs DE ANOTACIÓN
# ─────────────────────────────────────────────
print(f"\n[1/4] Leyendo anotaciones desde: {ANN_DIR}")

ann_files = sorted(ANN_DIR.glob("*.json"))
print(f"      Archivos .json encontrados: {len(ann_files)}")

records = []
errores = []

for ann_path in ann_files:
    # Nombre de imagen: quitar .json del final
    img_name = ann_path.stem   # p.ej. "ISIC_0024306.jpg"
    img_path = IMG_DIR / img_name

    try:
        with open(ann_path, "r") as f:
            ann = json.load(f)

        objects = ann.get("objects", [])
        height  = ann.get("size", {}).get("height", None)
        width   = ann.get("size", {}).get("width", None)

        if len(objects) == 0:
            errores.append((img_name, "sin_objetos"))
            continue

        # Tomamos la primera anotación (1 lesión por imagen en HAM10000)
        class_title = objects[0].get("classTitle", "unknown")
        class_code  = CLASS_CODE.get(class_title, "unknown")

        records.append({
            "image_name"  : img_name,
            "image_path"  : img_path.as_posix(),
            "class_title" : class_title,
            "class_code"  : class_code,
            "height"      : height,
            "width"       : width,
            "img_exists"  : img_path.exists(),
        })

    except Exception as e:
        errores.append((img_name, str(e)))

df = pd.DataFrame(records)
print(f"      Imágenes parseadas correctamente: {len(df)}")
if errores:
    print(f"      [AVISO] Errores / sin anotación: {len(errores)}")

# ─────────────────────────────────────────────
# 3. GUARDAR CSV MAESTRO
# ─────────────────────────────────────────────
csv_path = OUTPUT_DIR / "dataset_info.csv"
df.to_csv(csv_path, index=False)
print(f"\n[2/4] CSV maestro guardado en: {csv_path}")

# ─────────────────────────────────────────────
# 4. ESTADÍSTICAS BÁSICAS
# ─────────────────────────────────────────────
print("\n[3/4] Distribución de clases:")
print("-" * 45)

counts   = df["class_title"].value_counts()
total    = len(df)
majority = counts.iloc[0]

print(f"{'Clase':<35} {'N':>6}  {'%':>6}  {'Ratio':>8}")
print("-" * 45)
for cls, n in counts.items():
    pct   = 100 * n / total
    ratio = majority / n
    print(f"{cls:<35} {n:>6}  {pct:>5.1f}%  {ratio:>7.1f}x")
print("-" * 45)
print(f"{'TOTAL':<35} {total:>6}")

# Imágenes sin archivo físico
missing = df[~df["img_exists"]]
if len(missing) > 0:
    print(f"\n[AVISO] {len(missing)} imágenes referenciadas no existen en img/")
else:
    print(f"\n[OK] Todas las imágenes existen en ds/img/")

# ─────────────────────────────────────────────
# 5. GRÁFICOS
# ─────────────────────────────────────────────
print("\n[4/4] Generando gráficos...")

PALETTE = {
    "melanocytic nevi"             : "#4C72B0",
    "melanoma"                     : "#DD4949",
    "benign keratosis-like lesions": "#55A868",
    "basal cell carcinoma"         : "#C44E52",
    "actinic keratoses"            : "#8172B2",
    "vascular lesions"             : "#64B5CD",
    "dermatofibroma"               : "#CCB974",
}
colors_ordered = [PALETTE.get(c, "#999") for c in counts.index]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("HAM10000 — Análisis exploratorio de clases", fontsize=14, fontweight="bold")

# ── Gráfico 1: barras de frecuencia absoluta
ax1 = axes[0]
bars = ax1.barh(counts.index[::-1], counts.values[::-1],
                color=colors_ordered[::-1], edgecolor="white", height=0.6)
ax1.set_xlabel("Número de imágenes")
ax1.set_title("Distribución absoluta")
for bar, val in zip(bars, counts.values[::-1]):
    ax1.text(bar.get_width() + 30, bar.get_y() + bar.get_height() / 2,
             str(val), va="center", fontsize=9)
ax1.set_xlim(0, counts.max() * 1.15)
ax1.invert_xaxis()
ax1.yaxis.set_label_position("right")
ax1.yaxis.tick_right()

# ── Gráfico 2: tarta de proporciones
ax2 = axes[1]
wedges, texts, autotexts = ax2.pie(
    counts.values,
    labels=None,
    autopct="%1.1f%%",
    colors=colors_ordered,
    startangle=140,
    pctdistance=0.75,
    wedgeprops=dict(edgecolor="white", linewidth=1.5)
)
for at in autotexts:
    at.set_fontsize(8)
ax2.set_title("Proporción por clase")
legend_patches = [mpatches.Patch(color=colors_ordered[i], label=counts.index[i])
                  for i in range(len(counts))]
ax2.legend(handles=legend_patches, loc="lower center",
           bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=7.5)

# ── Gráfico 3: ratio de desbalanceo respecto a mayoría
ratios = majority / counts
ax3 = axes[2]
bars3 = ax3.bar(range(len(ratios)), ratios.values,
                color=colors_ordered, edgecolor="white")
ax3.axhline(1, color="black", linestyle="--", linewidth=0.8, label="Clase mayoritaria")
ax3.set_xticks(range(len(ratios)))
ax3.set_xticklabels([CLASS_CODE.get(c, c) for c in ratios.index],
                    rotation=30, ha="right", fontsize=9)
ax3.set_ylabel("Ratio de desbalanceo")
ax3.set_title("Desbalanceo por clase\n(veces menos que melanocytic nevi)")
for bar, val in zip(bars3, ratios.values):
    ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
             f"{val:.1f}x", ha="center", fontsize=8)
ax3.legend(fontsize=8)

plt.tight_layout()
plot_path = OUTPUT_DIR / "fase1_distribucion_clases.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"      Gráfico guardado en: {plot_path}")

# ─────────────────────────────────────────────
# 6. MUESTRA VISUAL DE IMÁGENES POR CLASE
# ─────────────────────────────────────────────
print("\n[BONUS] Generando muestra visual (1 imagen por clase)...")

fig2, axes2 = plt.subplots(1, len(counts), figsize=(20, 4))
fig2.suptitle("HAM10000 — Ejemplo de imagen por clase", fontsize=13, fontweight="bold")

for ax, (cls_name, _) in zip(axes2, counts.items()):
    sample_row = df[df["class_title"] == cls_name].iloc[0]
    img_p = Path(sample_row["image_path"])
    if img_p.exists():
        img = Image.open(img_p)
        ax.imshow(img)
    else:
        ax.text(0.5, 0.5, "No encontrada", ha="center", va="center")
    code = CLASS_CODE.get(cls_name, cls_name)
    ax.set_title(f"{code}\n({cls_name[:20]}...)" if len(cls_name) > 20
                 else f"{code}\n({cls_name})", fontsize=7.5)
    ax.axis("off")

plt.tight_layout()
sample_path = OUTPUT_DIR / "fase1_muestra_por_clase.png"
plt.savefig(sample_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"      Muestra guardada en: {sample_path}")

# ─────────────────────────────────────────────
# RESUMEN FINAL
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  RESUMEN FASE 1")
print("=" * 60)
print(f"  Total imágenes válidas : {total}")
print(f"  Clases encontradas     : {df['class_title'].nunique()}")
print(f"  Clase mayoritaria      : melanocytic nevi ({majority} imágenes)")
print(f"  Clase minoritaria      : {counts.index[-1]} ({counts.iloc[-1]} imágenes)")
print(f"  Ratio máx. desbalanceo : {majority / counts.iloc[-1]:.1f}x")
print(f"\n  Archivos generados en outputs/:")
print(f"    • dataset_info.csv")
print(f"    • fase1_distribucion_clases.png")
print(f"    • fase1_muestra_por_clase.png")
print("\n  → SIGUIENTE PASO: fase2_preprocesamiento.py")
print("=" * 60)
