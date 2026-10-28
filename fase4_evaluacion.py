"""
Phase 4 - Single evaluation on the held-out test set: metrics, confusion matrices and ROC curves.

TFM - Clasificación de Lesiones Cutáneas (HAM10000)
====================================================
FASE 4: Evaluación completa sobre conjunto TEST
         Métricas por clase, matrices de confusión, curvas ROC

Requisitos previos:
    - outputs/split_test.csv
    - outputs/class_weights.json
    - outputs/modelos/vgg16_best.pth
    - outputs/modelos/resnet50_best.pth
    - outputs/modelos/efficientnet_b3_best.pth

Ejecutar desde la carpeta raíz del proyecto:
    python fase4_evaluacion.py
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    classification_report, confusion_matrix,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────
OUTPUT_DIR  = Path("outputs")
MODELS_DIR  = OUTPUT_DIR / "modelos"
EVAL_DIR    = OUTPUT_DIR / "evaluacion"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

CSV_TEST    = OUTPUT_DIR / "split_test.csv"
CW_PATH     = OUTPUT_DIR / "class_weights.json"

IMG_SIZE    = 224
BATCH_SIZE  = 32
NUM_CLASSES = 7

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("  TFM - HAM10000  |  FASE 4: Evaluación sobre Test")
print("=" * 60)
print(f"\n  Dispositivo: {DEVICE}")

# ─────────────────────────────────────────────
# 1. CARGAR DATOS Y METADATOS
# ─────────────────────────────────────────────
print(f"\n[1/5] Cargando datos de test...")
df_test = pd.read_csv(CSV_TEST)

with open(CW_PATH) as f:
    cw_info = json.load(f)

LABEL_TO_CODE = {int(k): v for k, v in cw_info["label_to_code"].items()}
CODE_TO_NAME  = cw_info["code_to_name"]
LABEL_TO_NAME = {k: CODE_TO_NAME[v] for k, v in LABEL_TO_CODE.items()}

# Nombres cortos para los ejes de los gráficos
CLASS_NAMES_SHORT = [LABEL_TO_CODE[i] for i in range(NUM_CLASSES)]
CLASS_NAMES_FULL  = [LABEL_TO_NAME[i] for i in range(NUM_CLASSES)]

print(f"  Test: {len(df_test)} imágenes")
print(f"  Clases: {CLASS_NAMES_SHORT}")

# ─────────────────────────────────────────────
# 2. DATASET Y DATALOADER
# ─────────────────────────────────────────────
test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

class HAMDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df        = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        img   = Image.open(Path(str(row["image_path"]).replace("\\", "/"))).convert("RGB")
        label = int(row["label"])
        if self.transform:
            img = self.transform(img)
        return img, label

test_dataset = HAMDataset(df_test, transform=test_transform)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0, pin_memory=True)

# ─────────────────────────────────────────────
# 3. FUNCIÓN DE CARGA DE MODELOS
# ─────────────────────────────────────────────
def load_model(arch_name: str) -> nn.Module:
    if arch_name == "vgg16":
        model = models.vgg16(weights=None)
        model.classifier[6] = nn.Linear(
            model.classifier[6].in_features, NUM_CLASSES)
    elif arch_name == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    elif arch_name == "efficientnet_b3":
        model = models.efficientnet_b3(weights=None)
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features, NUM_CLASSES)

    weights_path = MODELS_DIR / f"{arch_name}_best.pth"
    state_dict   = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model

# ─────────────────────────────────────────────
# 4. INFERENCIA Y MÉTRICAS
# ─────────────────────────────────────────────
@torch.no_grad()
def predict(model, loader):
    all_preds, all_labels, all_probs = [], [], []
    for imgs, labels in loader:
        imgs   = imgs.to(DEVICE)
        logits = model(imgs)
        probs  = torch.softmax(logits, dim=1)
        preds  = probs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())
    return (np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs))

def compute_metrics(y_true, y_pred, y_prob, arch_name):
    acc  = accuracy_score(y_true, y_pred)
    f1m  = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    f1w  = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_true, y_pred,    average="macro", zero_division=0)

    # Por clase
    report = classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES_SHORT,
        output_dict=True,
        zero_division=0
    )

    print(f"\n  {'─'*50}")
    print(f"  {arch_name.upper()} — Métricas sobre TEST")
    print(f"  {'─'*50}")
    print(f"  Accuracy         : {acc:.4f}")
    print(f"  F1-macro         : {f1m:.4f}")
    print(f"  F1-weighted      : {f1w:.4f}")
    print(f"  Precision-macro  : {prec:.4f}")
    print(f"  Recall-macro     : {rec:.4f}")
    print(f"\n  Por clase:")
    print(f"  {'Clase':<8} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>9}")
    print(f"  {'─'*47}")
    for cls in CLASS_NAMES_SHORT:
        r = report[cls]
        print(f"  {cls:<8} {r['precision']:>10.4f} {r['recall']:>8.4f} "
              f"{r['f1-score']:>8.4f} {int(r['support']):>9}")

    return {
        "arch"      : arch_name,
        "accuracy"  : acc,
        "f1_macro"  : f1m,
        "f1_weighted": f1w,
        "precision" : prec,
        "recall"    : rec,
        "report"    : report,
    }

# ─────────────────────────────────────────────
# 5. EVALUAR LOS 3 MODELOS
# ─────────────────────────────────────────────
print(f"\n[2/5] Evaluando modelos sobre test set...")

ARCHITECTURES = ["vgg16", "resnet50", "efficientnet_b3"]
all_results   = {}

for arch in ARCHITECTURES:
    print(f"\n  Cargando {arch}...")
    model = load_model(arch)
    y_true, y_pred, y_prob = predict(model, test_loader)
    metrics = compute_metrics(y_true, y_pred, y_prob, arch)
    metrics["y_true"] = y_true
    metrics["y_pred"] = y_pred
    metrics["y_prob"] = y_prob
    all_results[arch] = metrics
    del model
    torch.cuda.empty_cache()

# ─────────────────────────────────────────────
# 6. MATRICES DE CONFUSIÓN
# ─────────────────────────────────────────────
print(f"\n[3/5] Generando matrices de confusión...")

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle("HAM10000 — Matrices de confusión (Test set)",
             fontsize=14, fontweight="bold")

for ax, arch in zip(axes, ARCHITECTURES):
    cm   = confusion_matrix(all_results[arch]["y_true"],
                            all_results[arch]["y_pred"])
    # Normalizar por fila (recall por clase)
    cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    sns.heatmap(cm_n, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=CLASS_NAMES_SHORT,
                yticklabels=CLASS_NAMES_SHORT,
                ax=ax, cbar=True, vmin=0, vmax=1,
                linewidths=0.5, linecolor="white")
    ax.set_title(f"{arch}\n(F1-macro={all_results[arch]['f1_macro']:.4f})",
                 fontsize=11)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)

plt.tight_layout()
cm_path = EVAL_DIR / "fase4_confusion_matrices.png"
plt.savefig(cm_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"  Guardado: {cm_path}")

# ─────────────────────────────────────────────
# 7. CURVAS ROC
# ─────────────────────────────────────────────
print(f"\n[4/5] Generando curvas ROC...")

fig2, axes2 = plt.subplots(1, 3, figsize=(20, 6))
fig2.suptitle("HAM10000 — Curvas ROC por clase (Test set)",
              fontsize=14, fontweight="bold")

COLORS = ["#4C72B0","#DD4949","#55A868","#C44E52","#8172B2","#64B5CD","#CCB974"]

for ax, arch in zip(axes2, ARCHITECTURES):
    y_true = all_results[arch]["y_true"]
    y_prob = all_results[arch]["y_prob"]
    y_bin  = label_binarize(y_true, classes=list(range(NUM_CLASSES)))

    auc_scores = []
    for i in range(NUM_CLASSES):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc     = auc(fpr, tpr)
        auc_scores.append(roc_auc)
        ax.plot(fpr, tpr, color=COLORS[i], lw=1.5,
                label=f"{CLASS_NAMES_SHORT[i]} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Random")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    mean_auc = np.mean(auc_scores)
    ax.set_title(f"{arch}\n(AUC medio={mean_auc:.4f})", fontsize=11)
    ax.legend(loc="lower right", fontsize=7.5)
    ax.grid(alpha=0.3)
    all_results[arch]["mean_auc"] = mean_auc

plt.tight_layout()
roc_path = EVAL_DIR / "fase4_curvas_roc.png"
plt.savefig(roc_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"  Guardado: {roc_path}")

# ─────────────────────────────────────────────
# 8. TABLA COMPARATIVA FINAL
# ─────────────────────────────────────────────
print(f"\n[5/5] Tabla comparativa final...")

fig3, ax3 = plt.subplots(figsize=(10, 3))
ax3.axis("off")

col_labels = ["Modelo", "Accuracy", "F1-macro", "F1-weighted",
              "Precision", "Recall", "AUC medio"]
rows = []
for arch in ARCHITECTURES:
    r = all_results[arch]
    rows.append([
        arch,
        f"{r['accuracy']:.4f}",
        f"{r['f1_macro']:.4f}",
        f"{r['f1_weighted']:.4f}",
        f"{r['precision']:.4f}",
        f"{r['recall']:.4f}",
        f"{r['mean_auc']:.4f}",
    ])

table = ax3.table(cellText=rows, colLabels=col_labels,
                  loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 2.0)

# Resaltar mejor valor por columna
for col_idx in range(1, len(col_labels)):
    vals    = [float(rows[i][col_idx]) for i in range(len(ARCHITECTURES))]
    best_i  = int(np.argmax(vals))
    cell    = table[best_i + 1, col_idx]
    cell.set_facecolor("#d4edda")
    cell.set_text_props(fontweight="bold")

ax3.set_title("Comparativa de modelos — Test set",
              fontsize=13, fontweight="bold", pad=20)
plt.tight_layout()
table_path = EVAL_DIR / "fase4_tabla_comparativa.png"
plt.savefig(table_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"  Guardado: {table_path}")

# ─────────────────────────────────────────────
# GUARDAR MÉTRICAS EN CSV
# ─────────────────────────────────────────────
summary_rows = []
for arch in ARCHITECTURES:
    r = all_results[arch]
    summary_rows.append({
        "modelo"      : arch,
        "accuracy"    : r["accuracy"],
        "f1_macro"    : r["f1_macro"],
        "f1_weighted" : r["f1_weighted"],
        "precision"   : r["precision"],
        "recall"      : r["recall"],
        "auc_medio"   : r["mean_auc"],
    })
df_summary = pd.DataFrame(summary_rows)
csv_path   = EVAL_DIR / "fase4_metricas_resumen.csv"
df_summary.to_csv(csv_path, index=False)

# ─────────────────────────────────────────────
# RESUMEN FINAL EN TERMINAL
# ─────────────────────────────────────────────
best_arch = max(all_results, key=lambda a: all_results[a]["f1_macro"])

print("\n" + "=" * 60)
print("  RESUMEN FASE 4 — EVALUACIÓN SOBRE TEST")
print("=" * 60)
print(f"\n  {'Modelo':<20} {'Acc':>7} {'F1-mac':>8} {'F1-w':>7} "
      f"{'Prec':>7} {'Rec':>7} {'AUC':>7}")
print(f"  {'─'*57}")
for arch in ARCHITECTURES:
    r = all_results[arch]
    marker = " ★" if arch == best_arch else ""
    print(f"  {arch:<20} {r['accuracy']:>7.4f} {r['f1_macro']:>8.4f} "
          f"{r['f1_weighted']:>7.4f} {r['precision']:>7.4f} "
          f"{r['recall']:>7.4f} {r['mean_auc']:>7.4f}{marker}")

print(f"\n  ★ Mejor modelo: {best_arch.upper()}")
print(f"\n  Archivos generados en outputs/evaluacion/:")
print(f"    • fase4_confusion_matrices.png")
print(f"    • fase4_curvas_roc.png")
print(f"    • fase4_tabla_comparativa.png")
print(f"    • fase4_metricas_resumen.csv")
print(f"\n  Pipeline completado.")
print("=" * 60)
