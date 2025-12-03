import os
import cv2
import numpy as np
from supabase import create_client, Client
from datetime import datetime
import uuid
from dotenv import load_dotenv


load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
BUCKET_NAME = "neurovitis"

def subir_imagen_cv2(imagen_cv2, carpeta="uploads"):
    try:
        _, buffer = cv2.imencode('.jpg', imagen_cv2)
        image_bytes = buffer.tobytes()
        
        filename = f"{carpeta}/{datetime.now().strftime('%Y%m%d')}_{uuid.uuid4()}.jpg"
        
        supabase.storage.from_(BUCKET_NAME).upload(
            path=filename,
            file=image_bytes,
            file_options={"content-type": "image/jpeg"}
        )
        
        public_url = supabase.storage.from_(BUCKET_NAME).get_public_url(filename)
        return public_url
        
    except Exception as e:
        print(f"Error subiendo imagen a Supabase: {e}")
        return None

def guardar_diagnostico(resultado_json, img_original_cv2, mask_disease_cv2):
    print("Iniciando persistencia en Supabase...")
    
    # 1. Subir Imagen Original
    url_original = subir_imagen_cv2(img_original_cv2, carpeta="originals")
    
    # 2. Subir Máscara (Si existe)
    url_mask = None
    if mask_disease_cv2 is not None:
        url_mask = subir_imagen_cv2(mask_disease_cv2, carpeta="masks")
    
    # 3. Insertar en Base de Datos
    data = {
        "disease": resultado_json['diagnosis']['disease'],
        "confidence": resultado_json['diagnosis']['confidence'],
        "fuzzy_probs": resultado_json['diagnosis']['fuzzy_probs'],
        
        "severity_label": resultado_json['expert_analysis']['severity_label'],
        "damage_percent": resultado_json['expert_analysis']['tissue_damage_percent'],
        "roi_detected": resultado_json['expert_analysis']['roi_detected'],
        
        "original_image_url": url_original,
        "mask_image_url": url_mask
    }
    
    try:
        response = supabase.table("diagnostics").insert(data).execute()
        print("Diagnóstico guardado en BD con éxito.")
        return response.data[0]['id'] # Retornamos el ID por si acaso
    except Exception as e:
        print(f"Error guardando en BD: {e}")
        return None

def obtener_resumen_historico(limite=10):
    if supabase is None: return []

    try:
        response = supabase.table("diagnostics")\
            .select("disease, confidence, severity_label, created_at")\
            .order("created_at", desc=True)\
            .limit(limite)\
            .execute()
            
        return response.data if response.data else []
    except Exception as e:
        print(f"Error leyendo historial: {e}")
        return []