import io

from flask import Flask, request, jsonify, send_from_directory
import base64, csv, datetime, os
from pathlib import Path
from PIL import Image
# Importa directamente tu script actual
from captura_y_predecir import (
    cargar_modelo, capturar_imagen, clasificar,
    enviar_tcp, TCP_PALABRAS, EMOJIS, CLASES
)

app = Flask(__name__, static_folder='web')
modelo = cargar_modelo()

DATASET_SAVE = Path(__file__).parent / "dataset_feedback"

def guardar_feedback(img_path: str, clase_correcta: str, origen: str):
    """origen = 'confirmado' | 'corregido'"""
    carpeta = DATASET_SAVE / clase_correcta
    carpeta.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    dest = carpeta / f"{ts}.jpg"
    # Mover la captura temporal al dataset
    import shutil
    shutil.copy(img_path, dest)
    # Log CSV
    log = DATASET_SAVE / "labels.csv"
    escribir_header = not log.exists()
    with open(log, 'a', newline='') as f:
        w = csv.writer(f)
        if escribir_header:
            w.writerow(['ruta', 'clase', 'origen', 'timestamp'])
        w.writerow([str(dest), clase_correcta, origen, ts])

@app.route('/capturar', methods=['POST'])
def capturar():
    """Dispara la PiCamera y devuelve la foto + predicción."""
    ruta = capturar_imagen()  # tu función, guarda en /tmp/ecobot_captura.jpg
    bote, confianza, prob_bote, _ = clasificar(modelo, ruta)

    with open(ruta, 'rb') as f:
        img_b64 = base64.b64encode(f.read()).decode()

    return jsonify({
        'imagen': f'data:image/jpeg;base64,{img_b64}',
        'bote': bote,
        'confianza': round(confianza * 100, 1),
        'desglose': {k: round(v*100,1) for k,v in prob_bote.items()},
        'emoji': EMOJIS[bote],
        'ruta_tmp': ruta   # la mantenemos para guardarla después
    })


# Agrega este endpoint en ecobot_server.py
# justo después del endpoint /capturar existente

@app.route('/capturar_imagen', methods=['POST'])
def capturar_imagen_upload():
    """Clasifica una imagen enviada desde la galería del celular (base64)."""
    data = request.json['imagen']  # data:image/jpeg;base64,...
    img_bytes = base64.b64decode(data.split(',')[1])

    # Guardar en /tmp igual que la cámara, para reutilizar la misma lógica
    ruta = "/tmp/ecobot_captura.jpg"
    with open(ruta, 'wb') as f:
        f.write(img_bytes)

    img = Image.open(io.BytesIO(img_bytes))
    bote, confianza, prob_bote, _ = clasificar(modelo, ruta)

    return jsonify({
        'imagen': data,
        'bote': bote,
        'confianza': round(confianza * 100, 1),
        'desglose': {k: round(v * 100, 1) for k, v in prob_bote.items()},
        'emoji': EMOJIS[bote],
        'ruta_tmp': ruta
    })

@app.route('/confirmar', methods=['POST'])
def confirmar():
    """Usuario confirma o corrige. Guarda foto y manda TCP al ESP32."""
    body = request.json
    clase_final = body['clase_correcta']   # bote, no clase interna
    origen      = body['origen']           # 'confirmado' o 'corregido'
    ruta_tmp    = body['ruta_tmp']

    guardar_feedback(ruta_tmp, clase_final, origen)
    enviar_tcp(TCP_PALABRAS[clase_final])  # tu función TCP existente

    return jsonify({'ok': True})

@app.route('/')
def index():
    return send_from_directory('../web', 'index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)