# src/v1/predecir.py
import torch
import sys
from pathlib import Path
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image

BASE = Path(__file__).parent.parent

CLASES = ['basura_general', 'carton', 'metal', 'organico', 'papel', 'plastico', 'vidrio']

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

DEVICE = torch.device("cpu")

if len(sys.argv) < 2:
    print("Uso: python predecir.py <ruta_imagen>")
    sys.exit(1)

# ── Carga modelo ──────────────────────────────────────────────────────────────
checkpoint = torch.load(BASE / "modelos" / "modelo_latest_colab.pth", map_location="cpu")

modelo = models.efficientnet_b0(weights=None)
modelo.classifier[1] = nn.Linear(modelo.classifier[1].in_features, len(CLASES))
modelo.load_state_dict(checkpoint["model_state"])
modelo.eval()

# ── Preprocesa imagen ─────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

ruta = sys.argv[1]
img    = Image.open(ruta).convert("RGB")
tensor = transform(img).unsqueeze(0).to(DEVICE)

# ── Predicción ────────────────────────────────────────────────────────────────
with torch.no_grad():
    probs = torch.softmax(modelo(tensor), dim=1)[0]

prob_clase = dict(zip(CLASES, probs.tolist()))

prob_bote = {
    bote: sum(prob_clase.get(c, 0.0) for c in clases)
    for bote, clases in GRUPOS.items()
}

bote      = max(prob_bote, key=prob_bote.get)
confianza = prob_bote[bote]

# ── Output ────────────────────────────────────────────────────────────────────
print(f"\nImagen : {ruta}")
print(f"\n{'═'*50}")
print(f"  {EMOJIS[bote]}  →  {confianza*100:.1f}% de confianza")
print(f"{'═'*50}\n")

print("Por bote:")
for nombre, prob in sorted(prob_bote.items(), key=lambda x: x[1], reverse=True):
    barra  = "█" * int(prob * 30)
    activo = " ←" if nombre == bote else ""
    print(f"  {EMOJIS[nombre]:25} {barra:30} {prob*100:.1f}%{activo}")

print("\nDesglose interno:")
for nombre, prob in sorted(prob_clase.items(), key=lambda x: x[1], reverse=True):
    barra = "█" * int(prob * 30)
    print(f"  {nombre:16} {barra:30} {prob*100:.1f}%")