# src/predecir.py
import torch
import sys
from pathlib import Path
from modelo import ClasificadorResiduos

BASE = Path(__file__).parent.parent

clf = ClasificadorResiduos()
checkpoint = torch.load(BASE / "modelos" / "modelo_latest.pth", map_location="cpu")
clf.modelo.load_state_dict(checkpoint["model_state"])

ruta = sys.argv[1]
bote, confianza, prob_bote, prob_clase = clf.predecir_bote(ruta)

print(f"\nImagen: {ruta}")
print(f"\n{'─'*45}")
print(f"  BOTE: {bote.upper()}  ({confianza*100:.1f}%)")
print(f"{'─'*45}\n")

print("Por bote (lo que importa):")
for nombre, prob in sorted(prob_bote.items(), key=lambda x: x[1], reverse=True):
    barra   = "█" * int(prob * 30)
    activo  = " ←" if nombre == bote else ""
    print(f"  {nombre:10} {barra:30} {prob*100:.1f}%{activo}")

print("\nDesglose interno:")
for nombre, prob in sorted(prob_clase.items(), key=lambda x: x[1], reverse=True):
    barra = "█" * int(prob * 30)
    print(f"  {nombre:10} {barra:30} {prob*100:.1f}%")