# src/finetune.py
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from pathlib import Path
from modelo import ClasificadorResiduos, CLASES, DEVICE
import time

BASE = Path(__file__).parent.parent

clf = ClasificadorResiduos(ruta_data=BASE / "data")
checkpoint = torch.load(BASE / "modelos" / "modelo_mejor.pth", map_location="cpu")
clf.modelo.load_state_dict(checkpoint["model_state"])
print("Modelo base cargado\n")

for nombre, param in clf.modelo.named_parameters():
    # Descongela: features.6, features.7, features.8 y el classifier
    if any(f"features.{i}" in nombre for i in [6, 7, 8]):
        param.requires_grad = True
    if "classifier" in nombre:
        param.requires_grad = True

# Verifica qué quedó descongelado
params_entrenables = sum(p.numel() for p in clf.modelo.parameters() if p.requires_grad)
params_totales     = sum(p.numel() for p in clf.modelo.parameters())
print(f"Parámetros entrenables: {params_entrenables:,} / {params_totales:,}")

# ── 3. Augmentation más agresivo para fondos variables ───────────────────────
transform_train = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),                          # recorte aleatorio
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),                       # más rotación
    transforms.ColorJitter(brightness=0.4, contrast=0.4,
                           saturation=0.3, hue=0.1),     # más variación de color
    transforms.RandomGrayscale(p=0.1),                   # ocasionalmente grises
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

# ── 4. Optimizador con learning rate MUY bajo ────────────────────────────────
# LR bajo = ajuste fino sin destruir lo que ya aprendió
optimizador = torch.optim.Adam([
    {"params": [p for n,p in clf.modelo.named_parameters()
                if "features" in n and p.requires_grad], "lr": 1e-5},
    {"params": clf.modelo.classifier.parameters(),       "lr": 1e-4},
])
criterio   = nn.CrossEntropyLoss()
scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizador, T_max=10)

# ── 5. Loop de fine-tuning ───────────────────────────────────────────────────
EPOCAS        = 10
mejor_val_acc = 0

print(f"\nFine-tuning por {EPOCAS} épocas\n")

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
    train_loss = loss_total / total
    train_acc  = correctas  / total

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
    val_loss = loss_total / total
    val_acc  = correctas  / total

    scheduler.step()

    if val_acc > mejor_val_acc:
        mejor_val_acc = val_acc
        torch.save({
            "model_state": clf.modelo.state_dict(),
            "clases":      CLASES,
        }, BASE / "modelos" / "modelo_final.pth")
        sufijo = " ← mejor"
    else:
        sufijo = ""

    print(f"Época {epoca:02}/{EPOCAS} | "
          f"train_loss: {train_loss:.4f}  train_acc: {train_acc:.3f} | "
          f"val_loss: {val_loss:.4f}  val_acc: {val_acc:.3f} | "
          f"{time.time()-t0:.1f}s{sufijo}")

print(f"\nMejor val_acc fine-tune: {mejor_val_acc:.3f}")
print(f"Modelo guardado en: modelos/modelo_final.pth")