# src/finetune_6clases.py
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime
from modelo import ClasificadorResiduos, CLASES, ALIAS, DEVICE
import time

BASE = Path(__file__).parent.parent

# ── 1. Carga el mejor modelo actual ──────────────────────────────────────────
clf = ClasificadorResiduos()
checkpoint = torch.load(BASE / "modelos" / "modelo_mejor.pth", map_location="cpu")
clf.modelo.load_state_dict(checkpoint["model_state"])
print(f"Modelo cargado — base: 84.9%\n")

# ── 2. Descongela últimas 3 capas del backbone ────────────────────────────────
for nombre, param in clf.modelo.named_parameters():
    if any(f"features.{i}" in nombre for i in [6, 7, 8]):
        param.requires_grad = True
    if "classifier" in nombre:
        param.requires_grad = True

params_entrenables = sum(p.numel() for p in clf.modelo.parameters() if p.requires_grad)
params_totales     = sum(p.numel() for p in clf.modelo.parameters())
print(f"Parámetros entrenables: {params_entrenables:,} / {params_totales:,}\n")

# ── 3. Augmentation agresivo ──────────────────────────────────────────────────
transform_train = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.4, contrast=0.4,
                           saturation=0.3, hue=0.1),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])
transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

train_ds = datasets.ImageFolder(BASE / "data/train", transform=transform_train)
val_ds   = datasets.ImageFolder(BASE / "data/val",   transform=transform_val)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False, num_workers=2)

# ── 4. Optimizador con LR diferenciado ───────────────────────────────────────
# Backbone descongelado → LR muy bajo para no destruir lo aprendido
# Classifier → LR un poco más alto
optimizador = torch.optim.Adam([
    {"params": [p for n, p in clf.modelo.named_parameters()
                if "features" in n and p.requires_grad], "lr": 1e-5},
    {"params": clf.modelo.classifier.parameters(),       "lr": 1e-4},
])
criterio  = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizador, T_max=10)

# ── 5. Loop ───────────────────────────────────────────────────────────────────
EPOCAS        = 10
mejor_val_acc = 0

print(f"Fine-tuning por {EPOCAS} épocas\n")

for epoca in range(1, EPOCAS + 1):
    t0 = time.time()

    # Train
    clf.modelo.train()
    loss_total, correctas, total = 0, 0, 0
    for imgs, etiquetas in train_loader:
        optimizador.zero_grad()
        salidas = clf.modelo(imgs)
        loss    = criterio(salidas, etiquetas)
        loss.backward()
        optimizador.step()
        loss_total += loss.item() * imgs.size(0)
        _, preds   = torch.max(salidas, 1)
        correctas  += (preds == etiquetas).sum().item()
        total      += imgs.size(0)
    train_loss, train_acc = loss_total / total, correctas / total

    # Val
    clf.modelo.eval()
    loss_total, correctas, total = 0, 0, 0
    with torch.no_grad():
        for imgs, etiquetas in val_loader:
            salidas = clf.modelo(imgs)
            loss    = criterio(salidas, etiquetas)
            loss_total += loss.item() * imgs.size(0)
            _, preds   = torch.max(salidas, 1)
            correctas  += (preds == etiquetas).sum().item()
            total      += imgs.size(0)
    val_loss, val_acc = loss_total / total, correctas / total

    scheduler.step()

    if val_acc > mejor_val_acc:
        mejor_val_acc = val_acc
        ts = datetime.now().strftime("%m%d_%H%M")
        ruta = BASE / "modelos" / f"modelo_ft6_{ts}_{val_acc:.3f}.pth"
        torch.save({
            "model_state": clf.modelo.state_dict(),
            "clases":      CLASES,
            "alias":       ALIAS,
        }, ruta)
        # También guarda como latest para uso fácil
        torch.save({
            "model_state": clf.modelo.state_dict(),
            "clases":      CLASES,
            "alias":       ALIAS,
        }, BASE / "modelos" / "modelo_latest.pth")
        sufijo = f" ← mejor (guardado)"
    else:
        sufijo = ""

    print(f"Época {epoca:02}/{EPOCAS} | "
          f"train_loss: {train_loss:.4f}  train_acc: {train_acc:.3f} | "
          f"val_loss: {val_loss:.4f}  val_acc: {val_acc:.3f} | "
          f"{time.time()-t0:.1f}s{sufijo}")

print(f"\nMejor val_acc fine-tune: {mejor_val_acc:.3f}")
print(f"Modelos guardados en: modelos/")