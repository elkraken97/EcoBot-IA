# verificar_entorno.py
import torch
import torchvision
import cv2
import PIL
import sklearn
import matplotlib

print(f"PyTorch:      {torch.__version__}")
print(f"Torchvision:  {torchvision.__version__}")
print(f"OpenCV:       {cv2.__version__}")
print(f"Pillow:       {PIL.__version__}")
print(f"scikit-learn: {sklearn.__version__}")
print(f"matplotlib:   {matplotlib.__version__}")
print(f"\nCUDA disponible: {torch.cuda.is_available()}")  # dirá False, es normal
print("Entorno listo.")