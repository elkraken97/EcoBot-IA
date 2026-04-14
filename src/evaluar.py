# src/evaluar.py
import torch
from modelo import ClasificadorResiduos

clf = ClasificadorResiduos(ruta_data="data")
clf.preparar_datos(batch_size=16)

# Carga el mejor modelo guardado

from pathlib import Path
BASE = Path(__file__).parent.parent
checkpoint = torch.load(BASE / "modelos" / "modelo_latest.pth", map_location="cpu")
clf.modelo.load_state_dict(checkpoint["model_state"])
print("Modelo cargado: modelo_latest.pth")

clf.evaluar()