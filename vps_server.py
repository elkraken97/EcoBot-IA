"""
vps_server.py — Servidor de clasificación EcoBot (corre en el VPS)
===================================================================
Recibe una imagen, la clasifica con EfficientNet-B0 y devuelve JSON.

Requisitos:
    pip install flask torch torchvision pillow

Estructura esperada en el VPS:
    /ruta/ecobot/
        vps_server.py
        modelos/
            modelo_latest.pth

Uso:
    python vps_server.py

Endpoint principal:
    POST /clasificar
    Body JSON: { "imagen": "data:image/jpeg;base64,..." }
    Respuesta: ver abajo
"""

import io
import base64
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from flask import Flask, request, jsonify

# ── Configuración ──────────────────────────────────────────────────────────────
BASE   = Path(__file__).parent
MODELO = BASE / "modelos" / "modelo_latest.pth"
DEVICE = torch.device("cpu")
PORT   = 5050   # ← cambia si quieres otro puerto

# ── Clases y grupos (igual que en la Raspberry) ────────────────────────────────
CLASES = ["basura_general", "carton", "metal", "organico", "papel", "plastico", "vidrio"]

GRUPOS = {
    "organico":       ["organico", "carton", "papel"],
    "metal":          ["metal"],
    "plastico":       ["plastico"],
    "basura_general": ["basura_general", "vidrio"],
}

EMOJIS = {
    "organico":       "🟤 Orgánico",
    "metal":          "⚙️  Metal",
    "plastico":       "🔵 Plástico",
    "basura_general": "🗑️  Basura general",
}

TCP_PALABRAS = {
    "organico":       "ORGANICO",
    "metal":          "METAL",
    "plastico":       "PLASTICO",
    "basura_general": "GENERAL",
}

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── Carga del modelo (una sola vez al arrancar) ────────────────────────────────
def cargar_modelo():
    log.info(f"Cargando modelo desde {MODELO} ...")
    ckpt   = torch.load(MODELO, map_location="cpu")
    modelo = models.efficientnet_b0(weights=None)
    modelo.classifier[1] = nn.Linear(
        modelo.classifier[1].in_features, len(CLASES)
    )
    modelo.load_state_dict(ckpt["model_state"])
    modelo.eval()
    log.info("Modelo listo.")
    return modelo

# ── Transformación de imagen ───────────────────────────────────────────────────
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                          [0.229, 0.224, 0.225]),
])

# ── Clasificación ──────────────────────────────────────────────────────────────
def clasificar(modelo, imagen_pil: Image.Image):
    tensor = TRANSFORM(imagen_pil).unsqueeze(0).to(DEVICE)

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

# ── App Flask ──────────────────────────────────────────────────────────────────
app   = Flask(__name__)
MODEL = cargar_modelo()

@app.route("/clasificar", methods=["POST"])
def clasificar_endpoint():
    """
    Body esperado (JSON):
        { "imagen": "data:image/jpeg;base64,/9j/..." }
        o bien solo el base64 puro sin el prefijo data:...

    Respuesta JSON:
    {
        "bote":      "metal",
        "palabra":   "METAL",
        "emoji":     "⚙️  Metal",
        "confianza": 94.3,
        "desglose": {
            "metal":          94.3,
            "plastico":        3.1,
            "organico":        1.8,
            "basura_general":  0.8
        },
        "detalle_clases": {
            "metal":          94.3,
            "plastico":        3.1,
            ...
        }
    }
    """
    # ── Extraer imagen ────────────────────────────────────────────────────────
    body = request.get_json(force=True, silent=True)
    if not body or "imagen" not in body:
        return jsonify({"error": "Se esperaba JSON con campo 'imagen' (base64)"}), 400

    raw = body["imagen"]
    if "," in raw:          # quita el prefijo data:image/jpeg;base64,
        raw = raw.split(",", 1)[1]

    try:
        img_bytes = base64.b64decode(raw)
        imagen    = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Imagen inválida: {e}"}), 400

    # ── Clasificar ────────────────────────────────────────────────────────────
    bote, confianza, prob_bote, prob_clase = clasificar(MODEL, imagen)

    log.info(f"→ {bote} ({confianza*100:.1f}%)")

    return jsonify({
        "bote":           bote,
        "palabra":        TCP_PALABRAS[bote],
        "emoji":          EMOJIS[bote],
        "confianza":      round(confianza * 100, 1),
        "desglose":       {k: round(v * 100, 1) for k, v in prob_bote.items()},
        "detalle_clases": {k: round(v * 100, 1) for k, v in prob_clase.items()},
    })

@app.route("/health", methods=["GET"])
def health():
    """Ping rápido para verificar que el servidor está vivo."""
    return jsonify({"ok": True, "clases": CLASES})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)