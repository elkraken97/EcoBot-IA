# src/exportar.py
import torch
from torchvision import transforms
from pathlib import Path
from modelo import ClasificadorResiduos, CLASES

BASE = Path(__file__).parent.parent

# ── 1. Carga el modelo final ─────────────────────────────────────────────────
clf = ClasificadorResiduos(ruta_data=BASE / "data")
checkpoint = torch.load(BASE / "modelos" / "modelo_final.pth", map_location="cpu")
clf.modelo.load_state_dict(checkpoint["model_state"])
clf.modelo.eval()
print("Modelo cargado correctamente")

# ── 2. Exporta a ONNX ────────────────────────────────────────────────────────
# ONNX necesita un tensor de ejemplo del mismo tamaño que usará en producción
tensor_ejemplo = torch.randn(1, 3, 224, 224)  # batch=1, RGB, 224x224

ruta_onnx = BASE / "modelos" / "modelo_final.onnx"

torch.onnx.export(
    clf.modelo,
    tensor_ejemplo,
    ruta_onnx,
    export_params=True,        # incluye los pesos dentro del archivo
    opset_version=18,          # versión estable y compatible
    input_names=["imagen"],
    output_names=["prediccion"],
    dynamic_axes={             # permite batch size variable (1 imagen o varias)
        "imagen":     {0: "batch"},
        "prediccion": {0: "batch"},
    }
)
print(f"Exportado: {ruta_onnx}")

# ── 3. Verifica que el archivo es válido ─────────────────────────────────────
import onnx
modelo_onnx = onnx.load(ruta_onnx)
onnx.checker.check_model(modelo_onnx)
print("Verificación ONNX: OK")

# ── 4. Prueba de inferencia con onnxruntime ──────────────────────────────────
import onnxruntime as ort
import numpy as np
from PIL import Image

sesion = ort.InferenceSession(str(ruta_onnx))

# Preprocesa una imagen de prueba igual que en entrenamiento
def preprocesar(ruta_img):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    img = Image.open(ruta_img).convert("RGB")
    return transform(img).unsqueeze(0).numpy()

# Prueba con el plátano difícil
ruta_prueba = "/tmp/8iik0tzzqbgf1.jpeg"
entrada = preprocesar(ruta_prueba)

salida = sesion.run(["prediccion"], {"imagen": entrada})[0]

# Softmax manual para obtener probabilidades
def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()

probs = softmax(salida[0])
idx   = probs.argmax()

print(f"\nPrueba de inferencia ONNX:")
print(f"Imagen:    {ruta_prueba}")
print(f"Resultado: {CLASES[idx].upper()} ({probs[idx]*100:.1f}%)")
print("\nTodas las probabilidades:")
for clase, prob in sorted(zip(CLASES, probs), key=lambda x: x[1], reverse=True):
    barra = "█" * int(prob * 30)
    print(f"  {clase:10} {barra:30} {prob*100:.1f}%")

# ── 5. Tamaño del archivo ────────────────────────────────────────────────────
mb = ruta_onnx.stat().st_size / 1024 / 1024
print(f"\nTamaño del modelo ONNX: {mb:.1f} MB")