# src/comparar_modelos.py
import torch
import sys
from pathlib import Path
from modelo import ClasificadorResiduos, CLASES

BASE = Path(__file__).parent.parent

def cargar(ruta_pth):
    clf = ClasificadorResiduos(ruta_data=BASE / "data")
    checkpoint = torch.load(ruta_pth, map_location="cpu")
    clf.modelo.load_state_dict(checkpoint["model_state"])
    return clf

def predecir_todas(clf, ruta_imagen):
    _, _, todas = clf.predecir(ruta_imagen)
    return todas

modelos = {
    "original  ": BASE / "modelos" / "modelo_mejor.pth",
    "fine-tune ": BASE / "modelos" / "modelo_final.pth",
}

ruta = sys.argv[1]
print(f"\nImagen: {ruta}\n")

for nombre, ruta_pth in modelos.items():
    clf   = cargar(ruta_pth)
    todas = predecir_todas(clf, ruta)
    ganador, prob = todas[0]
    print(f"── {nombre} → {ganador.upper()} {prob*100:.1f}%")
    for clase, p in todas:
        barra = "█" * int(p * 30)
        print(f"   {clase:10} {barra:30} {p*100:.1f}%")
    print()