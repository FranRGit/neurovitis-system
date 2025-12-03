from flask import Flask, request, jsonify
from core.ai_logic import NeuroVitisSystem
from core.agent_logic import neurobot_agente_mensaje
import os
from flask_cors import CORS 

app = Flask(__name__)
CORS(app)

# Configuración
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_PATH = os.path.join(BASE_DIR, 'models')

neuro_system = None

try:
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

@app.route('/chat', methods=['POST'])
def chat_agent():
    data = request.json
    
    # Validaciones
    if not data or 'message' not in data:
        return jsonify({"error": "Mensaje vacío"}), 400
    
    user_message = data['message']
    
    # Extraemos el contexto que envía Angular (puede venir vacío si no hay analisis aun)
    user_context = data.get('context', {}) 
    
    # Llamamos al agente pasando ambos datos
    respuesta = neurobot_agente_mensaje(user_message, user_context)
    
    return jsonify({
        "response": respuesta,
        "agent": "NeuroBot v1.0"
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=os.getenv("PORT"), debug=True)