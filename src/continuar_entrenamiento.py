# src/continuar_entrenamiento.py
import torch
from pathlib import Path
from modelo import ClasificadorResiduos

BASE = Path(__file__).parent.parent

clf = ClasificadorResiduos()
checkpoint = torch.load(BASE / "modelos" / "modelo_mejor.pth", map_location="cpu")
clf.modelo.load_state_dict(checkpoint["model_state"])
print("Continuando desde modelo_mejor.pth (84.5%)\n")

clf.preparar_datos(batch_size=16)
clf.entrenar(epocas=10)