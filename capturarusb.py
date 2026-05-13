#!/usr/bin/env python3
"""
capturar_foto_usb.py
--------------------
Monta una USB conectada a la Raspberry Pi, toma una foto con la cámara
y guarda la imagen en la USB. No borra ni modifica archivos existentes.

Uso:
    sudo python3 capturar_foto_usb.py

Requisitos:
    sudo apt install python3-picamera2   # Cámara del módulo oficial
    # ó
    sudo apt install fswebcam            # Para cámaras USB (webcam)
"""

import os
import sys
import subprocess
import time
from datetime import datetime
from pathlib import Path


# ─────────────────────────────────────────────
# CONFIGURACIÓN — ajusta según tu hardware
# ─────────────────────────────────────────────

# Punto de montaje donde se montará la USB
MOUNT_POINT = "/mnt/usb_ecobot"

# Carpeta dentro de la USB donde se guardarán las fotos
# (se crea si no existe, nunca borra otras carpetas)
CARPETA_FOTOS = "fotos_ecobot"

# Tipo de cámara: "picamera2" (módulo oficial) o "fswebcam" (cámara USB)
TIPO_CAMARA = "picamera2"

# Resolución de la foto
RESOLUCION = (1920, 1080)

# ─────────────────────────────────────────────


def encontrar_usb() -> str | None:
    """
    Detecta automáticamente el dispositivo USB conectado.
    Busca en /dev/sd* y /dev/mmcblk* excluyendo la tarjeta SD interna.
    Retorna la ruta del dispositivo (ej. '/dev/sda1') o None si no encuentra.
    """
    resultado = subprocess.run(
        ["lsblk", "-o", "NAME,TRAN,MOUNTPOINT", "--noheadings", "--list"],
        capture_output=True, text=True
    )

    for linea in resultado.stdout.splitlines():
        partes = linea.split()
        if len(partes) >= 2 and partes[1] == "usb":
            nombre = partes[0]
            # Preferir particiones (sda1, sda2...) sobre el disco base (sda)
            if any(c.isdigit() for c in nombre):
                return f"/dev/{nombre}"

    # Fallback: buscar /dev/sda1 directamente
    for candidato in ["/dev/sda1", "/dev/sdb1", "/dev/sdc1"]:
        if os.path.exists(candidato):
            return candidato

    return None


def montar_usb(dispositivo: str, punto_montaje: str) -> bool:
    """
    Monta el dispositivo USB en el punto de montaje indicado.
    Usa 'noatime' para no modificar timestamps de archivos existentes.
    Retorna True si el montaje fue exitoso.
    """
    # Verificar si ya está montada
    montajes = subprocess.run(
        ["mount"], capture_output=True, text=True
    ).stdout

    if dispositivo in montajes:
        print(f"  ✔ {dispositivo} ya está montado.")
        return True

    # Crear punto de montaje si no existe
    os.makedirs(punto_montaje, exist_ok=True)

    # Montar con opciones seguras (noatime = no modifica metadatos de archivos)
    cmd = ["mount", "-o", "noatime", dispositivo, punto_montaje]
    resultado = subprocess.run(cmd, capture_output=True, text=True)

    if resultado.returncode == 0:
        print(f"  ✔ USB montada en {punto_montaje}")
        return True
    else:
        # Intentar detectar el filesystem y montar con tipo explícito
        for fs in ["vfat", "exfat", "ntfs", "ext4"]:
            cmd_fs = ["mount", "-t", fs, "-o", "noatime", dispositivo, punto_montaje]
            res = subprocess.run(cmd_fs, capture_output=True, text=True)
            if res.returncode == 0:
                print(f"  ✔ USB montada en {punto_montaje} (filesystem: {fs})")
                return True

        print(f"  ✗ Error al montar: {resultado.stderr.strip()}")
        return False


def desmontar_usb(punto_montaje: str):
    """Desmonta la USB de forma segura."""
    # sync para asegurar que todos los datos fueron escritos
    subprocess.run(["sync"], check=True)
    resultado = subprocess.run(
        ["umount", punto_montaje], capture_output=True, text=True
    )
    if resultado.returncode == 0:
        print(f"  ✔ USB desmontada correctamente.")
    else:
        print(f"  ⚠ No se pudo desmontar: {resultado.stderr.strip()}")
        print("    Puedes desmontar manualmente con: sudo umount " + punto_montaje)


def tomar_foto_picamera2(ruta_salida: str) -> bool:
    """
    Toma una foto usando el módulo de cámara oficial de Raspberry Pi (picamera2).
    """
    try:
        from picamera2 import Picamera2
        cam = Picamera2()
        config = cam.create_still_configuration(
            main={"size": RESOLUCION}
        )
        cam.configure(config)
        cam.start()
        time.sleep(2)  # Dar tiempo al auto-exposición
        cam.capture_file(ruta_salida)
        cam.stop()
        cam.close()
        return True
    except ImportError:
        print("  ✗ picamera2 no instalada. Intenta: sudo apt install python3-picamera2")
        return False
    except Exception as e:
        print(f"  ✗ Error con picamera2: {e}")
        return False


def tomar_foto_fswebcam(ruta_salida: str) -> bool:
    """
    Toma una foto usando fswebcam (compatible con cámaras USB / webcams).
    """
    ancho, alto = RESOLUCION
    cmd = [
        "fswebcam",
        "-r", f"{ancho}x{alto}",
        "--no-banner",
        "-D", "2",          # delay de 2 seg para estabilizar exposición
        ruta_salida
    ]
    resultado = subprocess.run(cmd, capture_output=True, text=True)
    if resultado.returncode == 0:
        return True
    else:
        print(f"  ✗ Error con fswebcam: {resultado.stderr.strip()}")
        print("    Asegúrate de tener fswebcam: sudo apt install fswebcam")
        return False


def capturar_foto(directorio_destino: str) -> str | None:
    """
    Genera un nombre único con timestamp y toma la foto.
    Nunca sobrescribe fotos existentes.
    Retorna la ruta completa de la foto guardada, o None si falla.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre_archivo = f"foto_{timestamp}.jpg"
    ruta_completa = os.path.join(directorio_destino, nombre_archivo)

    # Por si acaso existe (milisegundos), agregar sufijo
    contador = 1
    while os.path.exists(ruta_completa):
        nombre_archivo = f"foto_{timestamp}_{contador}.jpg"
        ruta_completa = os.path.join(directorio_destino, nombre_archivo)
        contador += 1

    print(f"  📸 Capturando imagen: {nombre_archivo}")

    if TIPO_CAMARA == "picamera2":
        exito = tomar_foto_picamera2(ruta_completa)
    elif TIPO_CAMARA == "fswebcam":
        exito = tomar_foto_fswebcam(ruta_completa)
    else:
        print(f"  ✗ TIPO_CAMARA '{TIPO_CAMARA}' no reconocido. Usa 'picamera2' o 'fswebcam'.")
        return None

    return ruta_completa if exito else None


def verificar_root():
    """Verifica que el script se ejecute como root (necesario para montar)."""
    if os.geteuid() != 0:
        print("✗ Este script necesita permisos de superusuario para montar la USB.")
        print("  Ejecútalo con: sudo python3 capturar_foto_usb.py")
        sys.exit(1)


# ─────────────────────────────────────────────
# FLUJO PRINCIPAL
# ─────────────────────────────────────────────

def main():
    print("\n╔══════════════════════════════════════╗")
    print("║   EcoBot — Captura y guarda en USB   ║")
    print("╚══════════════════════════════════════╝\n")

    # 0. Verificar permisos
    verificar_root()

    # 1. Detectar USB
    print("🔍 Buscando USB conectada...")
    dispositivo = encontrar_usb()
    if not dispositivo:
        print("  ✗ No se encontró ninguna USB. Verifica la conexión.")
        sys.exit(1)
    print(f"  ✔ Dispositivo encontrado: {dispositivo}")

    # 2. Montar USB
    print(f"\n💾 Montando USB...")
    if not montar_usb(dispositivo, MOUNT_POINT):
        print("  ✗ No se pudo montar la USB.")
        sys.exit(1)

    try:
        # 3. Verificar espacio disponible
        stat = os.statvfs(MOUNT_POINT)
        espacio_mb = (stat.f_bavail * stat.f_frsize) / (1024 * 1024)
        print(f"  ℹ Espacio disponible en USB: {espacio_mb:.1f} MB")
        if espacio_mb < 5:
            print("  ✗ Espacio insuficiente en la USB (< 5 MB).")
            sys.exit(1)

        # 4. Crear carpeta de fotos (sin tocar lo demás)
        directorio_fotos = os.path.join(MOUNT_POINT, CARPETA_FOTOS)
        os.makedirs(directorio_fotos, exist_ok=True)
        print(f"  ✔ Carpeta de destino: {directorio_fotos}")

        # 5. Tomar la foto
        print(f"\n📷 Iniciando cámara ({TIPO_CAMARA})...")
        ruta_foto = capturar_foto(directorio_fotos)

        if ruta_foto and os.path.exists(ruta_foto):
            tamanio_kb = os.path.getsize(ruta_foto) / 1024
            print(f"\n✅ ¡Foto guardada exitosamente!")
            print(f"   Ruta  : {ruta_foto}")
            print(f"   Tamaño: {tamanio_kb:.1f} KB")
        else:
            print("\n✗ No se pudo guardar la foto.")
            sys.exit(1)

    finally:
        # 6. Desmontar USB siempre (incluso si hay error)
        print(f"\n🔌 Desmontando USB de forma segura...")
        desmontar_usb(MOUNT_POINT)

    print("\n✔ Proceso completado.\n")


if __name__ == "__main__":
    main()