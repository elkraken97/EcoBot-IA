# src/preparar_dataset.py
import glob
import shutil
import random
from pathlib import Path

random.seed(42)

FUENTE = Path("data_raw/standardized_256")
DEST   = Path("data")

MAPEO = {
    "glass":      "vidrio",
    "plastic":    "plastico",
    "biological": "organico",
    "cardboard":  "carton",
    "paper":      "papel",
    "metal":      "metal",
}

SPLIT_VAL = 0.2

def preparar():
    if DEST.exists():
        shutil.rmtree(DEST)
        print("data/ anterior eliminada\n")

    print(f"Leyendo imágenes desde: {FUENTE}\n")

    for carpeta_original, clase in MAPEO.items():
        carpeta = FUENTE / carpeta_original

        imagenes = list(carpeta.glob("*.jpg"))
        imagenes += list(carpeta.glob("*.png"))
        imagenes += list(carpeta.glob("*.jpeg"))

        if not imagenes:
            print(f"  ADVERTENCIA: no se encontraron imágenes en '{carpeta}'")
            continue

        random.shuffle(imagenes)
        corte      = int(len(imagenes) * (1 - SPLIT_VAL))
        train_imgs = imagenes[:corte]
        val_imgs   = imagenes[corte:]

        for split, imgs in [("train", train_imgs), ("val", val_imgs)]:
            destino = DEST / split / clase
            destino.mkdir(parents=True, exist_ok=True)
            for img in imgs:
                shutil.copy(img, destino / f"{carpeta_original}_{img.name}")

        print(f"  {carpeta_original:12} → {clase:10} | train: {len(train_imgs):4}  val: {len(val_imgs):4}")

    print("\nDataset listo en /data")

if __name__ == "__main__":
    preparar()