# capturar_y_predecir.py
"""
Captura una imagen con la cámara y clasifica el residuo.
Diseñado para Raspberry Pi 5 + IMX500.

Modos de uso:
  python capturar_y_predecir.py              → captura con cámara
  python capturar_y_predecir.py imagen.jpg   → usa imagen existente (pruebas)

Output por consola (legible por Arduino via serial):
  BOTE:metal
  CONFIANZA:98.2
"""

import sys
import torch
import torch.nn as nn
from torchvision import transforms, models
from pathlib import Path
from PIL import Image

BASE   = Path(__file__).parent
MODELO = BASE / "modelos" / "modelo_latest.pth"
DEVICE = torch.device("cpu")

CLASES = ["basura_general", "carton", "metal", "organico", "papel", "plastico", "vidrio"]

GRUPOS = {
    "organico":       ["organico", "carton", "papel"],
    "metal":          ["metal"],
    "plastico":       ["plastico"],
    "vidrio":         ["vidrio"],
    "basura_general": ["basura_general"],
}

EMOJIS = {
    "organico":       "🟤 Orgánico",
    "metal":          "⚙️  Metal",
    "plastico":       "🔵 Plástico",
    "vidrio":         "🟦 Vidrio",
    "basura_general": "🗑️  Basura general",
}

# ── Carga modelo ──────────────────────────────────────────────────────────────
def cargar_modelo():
    ckpt   = torch.load(MODELO, map_location="cpu")
    modelo = models.efficientnet_b0(weights=None)
    modelo.classifier[1] = nn.Linear(modelo.classifier[1].in_features, len(CLASES))
    modelo.load_state_dict(ckpt["model_state"])
    modelo.eval()
    return modelo

# ── Captura con cámara ────────────────────────────────────────────────────────
def capturar_imagen(ruta_tmp="/tmp/ecobot_captura.jpg"):
    try:
        from picamera2 import Picamera2
        cam = Picamera2()
        cam.configure(cam.create_still_configuration(
            main={"size": (640, 480)}
        ))
        cam.start()
        cam.capture_file(ruta_tmp)
        cam.stop()
        cam.close()
        return ruta_tmp
    except Exception as e:
        print(f"ERROR_CAMARA:{e}")
        sys.exit(1)

# ── Clasificación ─────────────────────────────────────────────────────────────
def clasificar(modelo, ruta_imagen):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                              [0.229, 0.224, 0.225]),
    ])
    img    = Image.open(ruta_imagen).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = torch.softmax(modelo(tensor), dim=1)[0]

    prob_clase = dict(zip(CLASES, probs.tolist()))
    prob_bote  = {
        bote: sum(prob_clase.get(c, 0.0) for c in clases)
        for bote, clases in GRUPOS.items()
    }

    bote      = max(prob_bote, key=prob_bote.get)
    confianza = prob_bote[bote]
    return bote, confianza, prob_bote, prob_clase

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    modelo = cargar_modelo()

    # Modo prueba: pasa ruta como argumento
    # Modo producción: captura con cámara
    if len(sys.argv) > 1:
        ruta = sys.argv[1]
    else:
        ruta = capturar_imagen()

    bote, confianza, prob_bote, prob_clase = clasificar(modelo, ruta)

    # ── Output para Arduino (siempre primero, limpio y parseable) ────────────
    print(f"BOTE:{bote}")
    print(f"CONFIANZA:{confianza*100:.1f}")

    # ── Output legible para humanos ──────────────────────────────────────────
    print(f"\n{'═'*45}")
    print(f"  {EMOJIS[bote]}  →  {confianza*100:.1f}%")
    print(f"{'═'*45}\n")

    print("Por bote:")
    for nombre, prob in sorted(prob_bote.items(), key=lambda x: x[1], reverse=True):
        barra  = "█" * int(prob * 30)
        activo = " ←" if nombre == bote else ""
        print(f"  {EMOJIS[nombre]:25} {barra:30} {prob*100:.1f}%{activo}")

    print("\nDesglose interno:")
    for nombre, prob in sorted(prob_clase.items(), key=lambda x: x[1], reverse=True):
        barra = "█" * int(prob * 30)
        print(f"  {nombre:16} {barra:30} {prob*100:.1f}%")