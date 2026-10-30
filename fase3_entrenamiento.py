"""
TFM - Clasificación de Lesiones Cutáneas (HAM10000)
====================================================
FASE 3: Entrenamiento — VGG16 / ResNet50 / EfficientNetB3
         Transfer learning + fine-tuning + class weights

Requisitos previos:
    - outputs/split_train.csv
    - outputs/split_val.csv
    - outputs/class_weights.json

Ejecutar desde la carpeta raíz del proyecto:
    python fase3_entrenamiento.py

Tiempo estimado: ~1.5-2h en RTX 3050
"""

import os, json, time, copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import f1_score, classification_report

# Configuración
OUTPUT_DIR   = Path("outputs")
MODELS_DIR   = OUTPUT_DIR / "modelos"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

CSV_TRAIN    = OUTPUT_DIR / "split_train.csv"
CSV_VAL      = OUTPUT_DIR / "split_val.csv"
CW_PATH      = OUTPUT_DIR / "class_weights.json"

IMG_SIZE     = 224        # estándar ImageNet
BATCH_SIZE   = 32
NUM_EPOCHS   = 30         # con early stopping, raramente llega al final
LR_HEAD      = 1e-3       # learning rate para la cabeza nueva
LR_FINETUNE  = 1e-5       # learning rate para fine-tuning del backbone (conservador)
PATIENCE     = 7          # early stopping
NUM_CLASSES  = 7
RANDOM_SEED  = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("  TFM - HAM10000  |  FASE 3: Entrenamiento")
print(f"\n  Dispositivo : {DEVICE}")
if DEVICE.type == "cuda":
    print(f"  GPU         : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM        : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# 1. Cargar datos y class weights
print(f"\nCargando datos...")
df_train = pd.read_csv(CSV_TRAIN)
df_val   = pd.read_csv(CSV_VAL)

with open(CW_PATH) as f:
    cw_info = json.load(f)

cw_dict  = {int(k): v for k, v in cw_info["class_weights"].items()}
cw_tensor = torch.tensor(
    [cw_dict[i] for i in range(NUM_CLASSES)], dtype=torch.float
).to(DEVICE)

print(f"  Train: {len(df_train)} | Val: {len(df_val)}")
print(f"  Class weights cargados: {[f'{cw_dict[i]:.3f}' for i in range(NUM_CLASSES)]}")

# 2. Dataset y transforms

# Transforms para train: augmentation
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2,
                           saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],   # ImageNet stats
                         std=[0.229, 0.224, 0.225]),
])

# Transforms para val/test: solo resize + normalize
val_transform = transforms.Compose([
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


train_dataset = HAMDataset(df_train, transform=train_transform)
val_dataset   = HAMDataset(df_val,   transform=val_transform)

train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                           shuffle=True,  num_workers=0, pin_memory=True)
val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                           shuffle=False, num_workers=0, pin_memory=True)

print(f"\nDataLoaders listos")
print(f"  Batches train: {len(train_loader)} | Batches val: {len(val_loader)}")

# 3. Funciones de entrenamiento

def build_model(arch_name: str) -> nn.Module:
    """Construye el modelo con cabeza nueva para NUM_CLASSES."""
    if arch_name == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        # Congelar backbone
        for p in model.features.parameters():
            p.requires_grad = False
        # Reemplazar clasificador
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, NUM_CLASSES)

    elif arch_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        for p in model.parameters():
            p.requires_grad = False
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, NUM_CLASSES)

    elif arch_name == "efficientnet_b3":
        model = models.efficientnet_b3(
            weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1
        )
        for p in model.parameters():
            p.requires_grad = False
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)

    return model.to(DEVICE)


def unfreeze_backbone(model, arch_name: str, lr_finetune: float):
    """Descongela el backbone para fine-tuning tras warm-up."""
    for p in model.parameters():
        p.requires_grad = True
    # Optimizador con lr diferenciado: backbone más bajo, cabeza más alto
    if arch_name == "vgg16":
        head_params    = list(model.classifier.parameters())
        backbone_params = list(model.features.parameters())
    elif arch_name == "resnet50":
        head_params    = list(model.fc.parameters())
        backbone_params = [p for n, p in model.named_parameters()
                           if "fc" not in n]
    elif arch_name == "efficientnet_b3":
        head_params    = list(model.classifier.parameters())
        backbone_params = list(model.features.parameters())

    optimizer = optim.Adam([
        {"params": backbone_params, "lr": lr_finetune},
        {"params": head_params,     "lr": lr_finetune * 5},
    ])
    return optimizer


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        preds      = outputs.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        loss    = criterion(outputs, labels)
        total_loss += loss.item() * imgs.size(0)
        preds       = outputs.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return total_loss / total, correct / total, f1


def train_model(arch_name: str):
    print(f"\n{'='*60}")
    print(f"  Entrenando: {arch_name.upper()}")
    print(f"{'='*60}")

    model     = build_model(arch_name)
    criterion = nn.CrossEntropyLoss(weight=cw_tensor)

    # FASE A: warm-up (solo cabeza, backbone congelado)
    if arch_name == "vgg16":
        optimizer = optim.Adam(model.classifier[6].parameters(), lr=LR_HEAD)
    elif arch_name == "resnet50":
        optimizer = optim.Adam(model.fc.parameters(), lr=LR_HEAD)
    elif arch_name == "efficientnet_b3":
        optimizer = optim.Adam(model.classifier[1].parameters(), lr=LR_HEAD)

    scheduler    = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    best_f1      = 0.0
    best_weights = None
    patience_cnt = 0
    history      = {"train_loss": [], "val_loss": [],
                    "train_acc": [],  "val_acc": [], "val_f1": []}

    WARMUP_EPOCHS = 5
    finetuning    = False

    for epoch in range(1, NUM_EPOCHS + 1):

        # Cambiar a fine-tuning tras warm-up
        if epoch == WARMUP_EPOCHS + 1 and not finetuning:
            print(f"\n  >> Epoch {epoch}: activando fine-tuning del backbone")
            optimizer  = unfreeze_backbone(model, arch_name, LR_FINETUNE)
            scheduler  = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, patience=3
            )
            finetuning = True

        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_acc, vl_f1 = evaluate(model, val_loader, criterion)
        elapsed = time.time() - t0

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)
        history["val_f1"].append(vl_f1)

        scheduler.step(vl_f1)

        marker = ""
        if vl_f1 > best_f1:
            best_f1      = vl_f1
            best_weights = copy.deepcopy(model.state_dict())
            patience_cnt = 0
            marker       = "  ← mejor"
        else:
            patience_cnt += 1

        phase = "FT" if finetuning else "WU"
        print(f"  [{phase}] Epoch {epoch:02d}/{NUM_EPOCHS} | "
              f"Loss {tr_loss:.4f}/{vl_loss:.4f} | "
              f"Acc {tr_acc:.3f}/{vl_acc:.3f} | "
              f"F1-macro {vl_f1:.4f} | "
              f"{elapsed:.0f}s{marker}")

        if patience_cnt >= PATIENCE:
            print(f"\n  Early stopping en epoch {epoch} (patience={PATIENCE})")
            break

    # Guardar mejor modelo (con manejo robusto de errores de escritura)
    model_path = MODELS_DIR / f"{arch_name}_best.pth"
    tmp_path   = MODELS_DIR / f"{arch_name}_tmp.pth"
    try:
        torch.save(best_weights, tmp_path)
        if tmp_path.exists():
            tmp_path.replace(model_path)
        print(f"\n  Mejor F1-macro val: {best_f1:.4f}")
        print(f"  Modelo guardado en: {model_path}")
    except Exception as e:
        print(f"\n  [ERROR] No se pudo guardar el modelo: {e}")
        print(f"  Continuando con el siguiente modelo...")

    return history, best_f1, model


# 4. Entrenar los 3 modelos
print(f"\nIniciando entrenamiento de 3 modelos...")
print(f"  Epochs máx: {NUM_EPOCHS} | Batch: {BATCH_SIZE} | Patience: {PATIENCE}")

ARCHITECTURES = ["vgg16", "resnet50", "efficientnet_b3"]
results       = {}

for arch in ARCHITECTURES:
    history, best_f1, model = train_model(arch)
    results[arch] = {"history": history, "best_f1": best_f1, "model": model}

# 5. Gráficos de curvas de entrenamiento
print(f"\nGenerando gráficos de entrenamiento...")

fig, axes = plt.subplots(3, 3, figsize=(18, 14))
fig.suptitle("HAM10000 — Curvas de entrenamiento por modelo",
             fontsize=14, fontweight="bold")

for row, arch in enumerate(ARCHITECTURES):
    h = results[arch]["history"]
    epochs = range(1, len(h["train_loss"]) + 1)

    # Loss
    axes[row, 0].plot(epochs, h["train_loss"], label="Train", color="#4C72B0")
    axes[row, 0].plot(epochs, h["val_loss"],   label="Val",   color="#DD4949")
    axes[row, 0].set_title(f"{arch} — Loss")
    axes[row, 0].set_xlabel("Epoch"); axes[row, 0].set_ylabel("Loss")
    axes[row, 0].legend(); axes[row, 0].grid(alpha=0.3)

    # Accuracy
    axes[row, 1].plot(epochs, h["train_acc"], label="Train", color="#4C72B0")
    axes[row, 1].plot(epochs, h["val_acc"],   label="Val",   color="#DD4949")
    axes[row, 1].set_title(f"{arch} — Accuracy")
    axes[row, 1].set_xlabel("Epoch"); axes[row, 1].set_ylabel("Accuracy")
    axes[row, 1].legend(); axes[row, 1].grid(alpha=0.3)

    # F1-macro
    axes[row, 2].plot(epochs, h["val_f1"], label="Val F1-macro", color="#55A868")
    axes[row, 2].set_title(f"{arch} — F1-macro Val")
    axes[row, 2].set_xlabel("Epoch"); axes[row, 2].set_ylabel("F1-macro")
    best_ep = int(np.argmax(h["val_f1"])) + 1
    axes[row, 2].axvline(best_ep, color="gray", linestyle="--",
                         label=f"Mejor epoch {best_ep}")
    axes[row, 2].legend(); axes[row, 2].grid(alpha=0.3)

plt.tight_layout()
plot_path = OUTPUT_DIR / "fase3_curvas_entrenamiento.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"  Guardado en: {plot_path}")

# Resumen comparativo
print("  RESUMEN FASE 3 — COMPARATIVA DE MODELOS (VAL)")
print(f"  {'Modelo':<20} {'Mejor F1-macro':>15}")
print(f"  {'-'*37}")
for arch in ARCHITECTURES:
    f1 = results[arch]["best_f1"]
    print(f"  {arch:<20} {f1:>15.4f}")

best_arch = max(results, key=lambda a: results[a]["best_f1"])
print(f"\n  Mejor modelo: {best_arch.upper()} "
      f"(F1-macro={results[best_arch]['best_f1']:.4f})")
for arch in ARCHITECTURES:
    print(f"    - {arch}_best.pth")
