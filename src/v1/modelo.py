# src/modelo.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from pathlib import Path
import time

# 6 clases internas — orden alfabético como ImageFolder las carga
CLASES = ["carton", "metal", "organico", "papel", "plastico", "vidrio"]

# Lo que ve el usuario final del bote
ALIAS = {
    "carton":   "organico",
    "papel":    "organico",
    "organico": "organico",
    "metal":    "metal",
    "plastico": "plastico",
    "vidrio":   "vidrio",
}

DEVICE = torch.device("cpu")
BASE   = Path(__file__).parent.parent


class ClasificadorResiduos:

    def __init__(self, ruta_data=None, lr=0.0003):
        self.ruta_data = Path(ruta_data) if ruta_data else BASE / "data"
        self.lr        = lr
        self.modelo    = self._construir_modelo()
        self.criterio  = nn.CrossEntropyLoss()
        self.optimizador = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.modelo.parameters()),
            lr=self.lr
        )
        self.historial = {"train_loss": [], "val_loss": [],
                          "train_acc":  [], "val_acc":  []}

    def _construir_modelo(self):
        modelo = models.efficientnet_b0(weights="IMAGENET1K_V1")
        for param in modelo.parameters():
            param.requires_grad = False
        in_features = modelo.classifier[1].in_features
        modelo.classifier[1] = nn.Linear(in_features, len(CLASES))
        return modelo.to(DEVICE)

    def preparar_datos(self, batch_size=16):
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

        train_ds = datasets.ImageFolder(self.ruta_data / "train",
                                        transform=transform_train)
        val_ds   = datasets.ImageFolder(self.ruta_data / "val",
                                        transform=transform_val)

        self.train_loader = DataLoader(train_ds, batch_size=16,
                                       shuffle=True,  num_workers=2)
        self.val_loader   = DataLoader(val_ds,   batch_size=16,
                                       shuffle=False, num_workers=2)

        print(f"Clases detectadas: {train_ds.classes}")
        print(f"Train: {len(train_ds)} imágenes | Val: {len(val_ds)} imágenes")

    def _epoch(self, loader, entrenando=True):
        self.modelo.train() if entrenando else self.modelo.eval()
        loss_total, correctas, total = 0, 0, 0

        with torch.set_grad_enabled(entrenando):
            for imgs, etiquetas in loader:
                imgs, etiquetas = imgs.to(DEVICE), etiquetas.to(DEVICE)
                if entrenando:
                    self.optimizador.zero_grad()
                salidas = self.modelo(imgs)
                loss    = self.criterio(salidas, etiquetas)
                if entrenando:
                    loss.backward()
                    self.optimizador.step()
                loss_total += loss.item() * imgs.size(0)
                _, predichas = torch.max(salidas, 1)
                correctas    += (predichas == etiquetas).sum().item()
                total        += imgs.size(0)

        return loss_total / total, correctas / total

    def entrenar(self, epocas=15):
        print(f"\nEntrenando por {epocas} épocas en {DEVICE}\n")
        mejor_val_acc = 0

        for epoca in range(1, epocas + 1):
            t0 = time.time()
            train_loss, train_acc = self._epoch(self.train_loader, entrenando=True)
            val_loss,   val_acc   = self._epoch(self.val_loader,   entrenando=False)

            self.historial["train_loss"].append(train_loss)
            self.historial["val_loss"].append(val_loss)
            self.historial["train_acc"].append(train_acc)
            self.historial["val_acc"].append(val_acc)

            if val_acc > mejor_val_acc:
                mejor_val_acc = val_acc
                self.guardar("modelo_mejor.pth")
                sufijo = " ← mejor"
            else:
                sufijo = ""

            print(f"Época {epoca:02}/{epocas} | "
                  f"train_loss: {train_loss:.4f}  train_acc: {train_acc:.3f} | "
                  f"val_loss: {val_loss:.4f}  val_acc: {val_acc:.3f} | "
                  f"{time.time()-t0:.1f}s{sufijo}")

        print(f"\nMejor val_acc: {mejor_val_acc:.3f}")

    def guardar(self, nombre="modelo.pth"):
        (BASE / "modelos").mkdir(exist_ok=True)
        torch.save({
            "model_state": self.modelo.state_dict(),
            "clases":      CLASES,
            "alias":       ALIAS,
            "lr":          self.lr,
        }, BASE / "modelos" / nombre)

    def predecir(self, ruta_imagen):
        from PIL import Image
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225]),
        ])
        img    = Image.open(ruta_imagen).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(DEVICE)

        self.modelo.eval()
        with torch.no_grad():
            salida = self.modelo(tensor)
            probs  = torch.softmax(salida, dim=1)[0]
            idx    = probs.argmax().item()

        clase_interna = CLASES[idx]
        clase_usuario = ALIAS[clase_interna]
        todas = sorted(zip(CLASES, probs.tolist()), key=lambda x: x[1], reverse=True)
        return clase_interna, clase_usuario, probs[idx].item(), todas

    def evaluar(self):
        from sklearn.metrics import classification_report, confusion_matrix

        self.modelo.eval()
        todas_preds, todas_labels = [], []

        with torch.no_grad():
            for imgs, etiquetas in self.val_loader:
                salidas = self.modelo(imgs)
                _, preds = torch.max(salidas, 1)
                todas_preds.extend(preds.cpu().numpy())
                todas_labels.extend(etiquetas.numpy())

        print("\nReporte por clase (6 clases internas):")
        print(classification_report(todas_labels, todas_preds,
                                    target_names=CLASES, digits=3))
        print("Matriz de confusión:")
        print(f"{'':12}", end="")
        for c in CLASES:
            print(f"{c:10}", end="")
        print()
        cm = confusion_matrix(todas_labels, todas_preds)
        for i, fila in enumerate(cm):
            print(f"{CLASES[i]:12}", end="")
            for val in fila:
                print(f"{val:<10}", end="")
            print()

    # Agrega esto en ClasificadorResiduos, después de predecir()
    def predecir_bote(self, ruta_imagen):
        from PIL import Image

        # Grupos por bote — qué clases internas pertenecen a cada contenedor
        GRUPOS = {
            "organico": ["organico", "carton", "papel"],
            "metal": ["metal"],
            "plastico": ["plastico"],
            "vidrio": ["vidrio"],
        }

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
        img = Image.open(ruta_imagen).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(DEVICE)

        self.modelo.eval()
        with torch.no_grad():
            salida = self.modelo(tensor)
            probs = torch.softmax(salida, dim=1)[0]

        # Probabilidad por clase interna
        prob_por_clase = dict(zip(CLASES, probs.tolist()))

        # Suma por bote
        prob_por_bote = {}
        for bote, clases in GRUPOS.items():
            prob_por_bote[bote] = sum(prob_por_clase.get(c, 0) for c in clases)

        # Bote ganador
        bote_ganador = max(prob_por_bote, key=prob_por_bote.get)
        confianza = prob_por_bote[bote_ganador]

        return bote_ganador, confianza, prob_por_bote, prob_por_clase