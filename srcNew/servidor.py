# servidor_test.py
import socket

IP   = "0.0.0.0"  # escucha en todas las interfaces
PORT = 5000

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((IP, PORT))
    s.listen()
    print(f"Escuchando en puerto {PORT}...\n")

    while True:
        conn, addr = s.accept()
        with conn:
            data = conn.recv(1024)
            if data:
                print(f"[{addr[0]}] â†’ {data.decode('utf-8')}")