# test_tcp.py
"""
Manda palabras aleatorias al servidor TCP para probar la conexión.
Uso: python test_tcp.py
"""

import socket
import random
import time

TCP_IP   = "192.168.0.9"  # ← pon la IP de la Raspberry aquí
TCP_PORT = 5000

PALABRAS = ["ORGANICO", "METAL", "GENERAL", "PLASTICO", "VIDRIO"]

def enviar_tcp(mensaje: str):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(5)
            s.connect((TCP_IP, TCP_PORT))
            s.sendall(mensaje.encode("utf-8"))
        print(f"[OK]    Enviado → {mensaje}")
    except Exception as e:
        print(f"[ERROR] {e}")

if __name__ == "__main__":
    print(f"Enviando a {TCP_IP}:{TCP_PORT}\n")
    while True:
        palabra = random.choice(PALABRAS)
        enviar_tcp(palabra)
        time.sleep(2)  # espera 2 segundos entre envíos