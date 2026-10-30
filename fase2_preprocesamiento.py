"""
TFM - Clasificación de Lesiones Cutáneas (HAM10000)
====================================================
FASE 2: Preprocesamiento — Splits estratificados + verificación

Requisitos previos:
    - outputs/dataset_info.csv  (generado en Fase 1)

Ejecutar desde la carpeta raíz del proyecto:
    python fase2_preprocesamiento.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

# Configuración
OUTPUT_DIR  = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

CSV_IN      = OUTPUT_DIR / "dataset_info.csv"
CSV_TRAIN   = OUTPUT_DIR / "split_train.csv"
CSV_VAL     = OUTPUT_DIR / "split_val.csv"
CSV_TEST    = OUTPUT_DIR / "split_test.csv"

# Proporciones del split
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
RANDOM_SEED = 42

# Mapeo código → nombre completo (para gráficos)
CLASS_CODE = {
    "melanocytic nevi"             : "nv",
    "melanoma"                     : "mel",
    "benign keratosis-like lesions": "bkl",
    "basal cell carcinoma"         : "bcc",
    "actinic keratoses"            : "akiec",
    "vascular lesions"             : "vasc",
    "dermatofibroma"               : "df",
}
CODE_TO_IDX = {code: i for i, code in enumerate(sorted(set(CLASS_CODE.values())))}

print("  TFM - HAM10000  |  FASE 2: Preprocesamiento y Splits")

# 1. Cargar CSV maestro
print(f"\nCargando CSV maestro: {CSV_IN}")
df = pd.read_csv(CSV_IN)
print(f"      Total imágenes: {len(df)}")

# Añadir columnas útiles
df["class_code"] = df["class_title"].map(CLASS_CODE)
df["label"]      = df["class_code"].map(CODE_TO_IDX)

print(f"      Clases encontradas: {df['class_code'].nunique()}")
print(f"      Índices de clase asignados:")
for name, code in CLASS_CODE.items():
    idx = CODE_TO_IDX[code]
    n   = len(df[df["class_code"] == code])
    print(f"         {idx}  {code:<6}  {name}  ({n} imgs)")

# 2. Split estratificado
print(f"\nGenerando splits estratificados "
      f"({int(TRAIN_RATIO*100)}/{int(VAL_RATIO*100)}/{int(TEST_RATIO*100)})...")

# Paso 1: separar test
df_trainval, df_test = train_test_split(
    df,
    test_size=TEST_RATIO,
    stratify=df["label"],
    random_state=RANDOM_SEED
)

# Paso 2: separar val del resto
val_ratio_adjusted = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
df_train, df_val = train_test_split(
    df_trainval,
    test_size=val_ratio_adjusted,
    stratify=df_trainval["label"],
    random_state=RANDOM_SEED
)

print(f"      Train : {len(df_train):>5} imágenes  ({100*len(df_train)/len(df):.1f}%)")
print(f"      Val   : {len(df_val):>5} imágenes  ({100*len(df_val)/len(df):.1f}%)")
print(f"      Test  : {len(df_test):>5} imágenes  ({100*len(df_test)/len(df):.1f}%)")

# 3. Guardar los CSV de cada split
print(f"\nGuardando CSVs de split...")
df_train.to_csv(CSV_TRAIN, index=False)
df_val.to_csv(CSV_VAL,     index=False)
df_test.to_csv(CSV_TEST,   index=False)
print(f"      {CSV_TRAIN}")
print(f"      {CSV_VAL}")
print(f"      {CSV_TEST}")

# 4. Verificar estratificación
print(f"\nVerificando estratificación por clase:")
print("-" * 65)
print(f"{'Clase':<8} {'Total':>6} {'Train':>7} {'Val':>7} {'Test':>7}  "
      f"{'Train%':>7} {'Val%':>6} {'Test%':>6}")
print("-" * 65)

for code in sorted(CODE_TO_IDX.keys()):
    n_total = len(df[df["class_code"] == code])
    n_train = len(df_train[df_train["class_code"] == code])
    n_val   = len(df_val[df_val["class_code"] == code])
    n_test  = len(df_test[df_test["class_code"] == code])
    print(f"{code:<8} {n_total:>6} {n_train:>7} {n_val:>7} {n_test:>7}  "
          f"{100*n_train/n_total:>6.1f}% {100*n_val/n_total:>5.1f}% "
          f"{100*n_test/n_total:>5.1f}%")
print("-" * 65)

# 5. Calcular class weights
print(f"\nCalculando class weights para el entrenamiento...")

from sklearn.utils.class_weight import compute_class_weight

classes    = np.array(sorted(CODE_TO_IDX.values()))
y_train    = df_train["label"].values
cw         = compute_class_weight("balanced", classes=classes, y=y_train)
cw_dict    = {int(c): float(w) for c, w in zip(classes, cw)}

# Guardar class weights
import json
cw_path = OUTPUT_DIR / "class_weights.json"
# Añadir también mapeo label→código para referencia
cw_info = {
    "class_weights": cw_dict,
    "label_to_code": {str(v): k for k, v in CODE_TO_IDX.items()},
    "code_to_name" : {v: k for k, v in CLASS_CODE.items()}
}
with open(cw_path, "w") as f:
    json.dump(cw_info, f, indent=2)

print(f"\n      Class weights (balanced):")
idx_to_code = {v: k for k, v in CODE_TO_IDX.items()}
for label, weight in sorted(cw_dict.items()):
    code = idx_to_code[label]
    name = {v: k for k, v in CLASS_CODE.items()}[code]
    n    = len(df_train[df_train["label"] == label])
    print(f"         label {label}  {code:<6}  weight={weight:.4f}  "
          f"(train imgs: {n})")
print(f"\n      Guardado en: {cw_path}")

# GRÁFICO: distribución en cada split
print(f"\nGenerando gráfico de verificación de splits...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
fig.suptitle("HAM10000 — Distribución por clase en cada split",
             fontsize=13, fontweight="bold")

splits     = [("Train (70%)", df_train), ("Val (15%)", df_val), ("Test (15%)", df_test)]
PALETTE    = ["#4C72B0","#DD4949","#55A868","#C44E52","#8172B2","#64B5CD","#CCB974"]
code_order = sorted(CODE_TO_IDX.keys())

for ax, (title, df_split) in zip(axes, splits):
    counts = df_split["class_code"].value_counts().reindex(code_order, fill_value=0)
    bars   = ax.bar(counts.index, counts.values, color=PALETTE, edgecolor="white")
    ax.set_title(f"{title}\n(n={len(df_split)})", fontsize=11)
    ax.set_xlabel("Clase")
    ax.set_ylabel("Imágenes")
    ax.set_xticklabels(counts.index, rotation=30, ha="right", fontsize=9)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                str(val), ha="center", fontsize=8)

plt.tight_layout()
plot_path = OUTPUT_DIR / "fase2_splits.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"      Guardado en: {plot_path}")

# Resumen final
print("  RESUMEN FASE 2")
print(f"  Split train/val/test : 70/15/15 estratificado")
print(f"  Train  : {len(df_train)} imágenes")
print(f"  Val    : {len(df_val)} imágenes")
print(f"  Test   : {len(df_test)} imágenes")
print(f"  Seed   : {RANDOM_SEED}")
print(f"\n  Class weight máximo (clase más rara)  : "
      f"{max(cw_dict.values()):.4f}")
print(f"  Class weight mínimo (clase más común) : "
      f"{min(cw_dict.values()):.4f}")
