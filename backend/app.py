from flask import Flask, request, jsonify
from core.ai_logic import NeuroVitisSystem
import os

app = Flask(__name__)

# Configuración
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_PATH = os.path.join(BASE_DIR, 'models')

neuro_system = None

try:
    if neuro_system is None:
        neuro_system = NeuroVitisSystem(MODELS_PATH)
except Exception as e:
    print(f"Advertencia: Modelos no cargados al inicio. {e}")


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No se envió ninguna imagen"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "Nombre de archivo vacío"}), 400

    try:
        # Llamamos a nuestro cerebro
        prediction = neuro_system.predict(file)
        return jsonify(prediction)
    
    except Exception as e:
        return jsonify({"error": f"Error interno: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "online", "system": "NeuroVitis Hybrid"}), 200

if __name__ == '__main__':
    # Ejecutar en el puerto 5000
    app.run(host='0.0.0.0', port=5000, debug=True)