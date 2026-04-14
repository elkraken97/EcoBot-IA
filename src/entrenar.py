# src/entrenar.py
from modelo import ClasificadorResiduos

clf = ClasificadorResiduos(ruta_data="../data", lr=0.001)
clf.preparar_datos(batch_size=16)
clf.entrenar(epocas=15)