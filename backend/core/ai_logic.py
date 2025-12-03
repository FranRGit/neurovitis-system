import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import joblib
import skfuzzy as fuzz
import os
import cv2
import base64
from PIL import Image
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler

# IMPORTANTE: Asegúrate de que estos archivos existan en backend/core/
from core import vision_logic
from core import expert_system

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        base_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.features = base_model.features
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.eval()

    def forward(self, x):
        x = self.features(x)
        x = self.pooling(x)
        x = torch.flatten(x, 1)
        return x

class NeuroVitisSystem:
    def __init__(self, models_dir):
        self.device = torch.device("cpu")
        self.models_dir = models_dir
        
        print(">>> [INIT] Cargando MobileNetV2...")
        self.vision_model = FeatureExtractor().to(self.device)
        
        # Transformaciones para MobileNet (Standard ImageNet)
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(">>> [INIT] Cargando Modelos Matemáticos (LDA + Fuzzy)...")        
        self._load_math_models()
        
    def _load_math_models(self):
        try:
            self.scaler = joblib.load(os.path.join(self.models_dir, 'modelo_scaler.pkl'))
            self.lda = joblib.load(os.path.join(self.models_dir, 'modelo_lda.pkl'))
            
            self.centers = np.load(os.path.join(self.models_dir, 'fcm_centers.npy'))
            self.mapping = joblib.load(os.path.join(self.models_dir, 'fcm_mapping.pkl'))
            
            self.n_clusters = self.centers.shape[0]
            self.m = 1.2 
        except Exception as e:
            print(f"Error fatal cargando modelos: {e}")
            raise e

    def _reorder_probabilities(self, u_raw):
        u_ordered = np.zeros(self.n_clusters)
        for cluster_id, real_label_id in self.mapping.items():
            u_ordered[real_label_id] += u_raw[cluster_id]
        return u_ordered

    def predict(self, image_file):
        # --- PASO 1: CARGA Y VISIÓN (OpenCV) ---       
        img_rgb = vision_logic.cargar_imagen_desde_memoria(image_file)
        if img_rgb is None:
            raise ValueError("El archivo no es una imagen válida o está corrupto.")
        
        # --- PASO 2: NORMALIZACIÓN (Dynamic ROI + CLAHE) ---
        roi_procesada = vision_logic.normalizar_hoja(img_rgb)
        
        # === CORRECCIÓN DE ARQUITECTO: FALLBACK LOGIC ===
        if roi_procesada is not None:
            # Escenario Ideal: Se detectó hoja, usamos el recorte limpio
            roi_final = roi_procesada
            roi_detected = True
        else:
            # Escenario Fallido: No se detectó hoja (pared, oscuridad)
            # En vez de crashear, usamos la imagen original redimensionada
            print("Advertencia: No se detectó ROI. Usando imagen completa.")
            roi_final = cv2.resize(img_rgb, (224, 224))
            roi_detected = False

        # --- PASO 3: SISTEMA EXPERTO (Severidad) ---
        # Analiza el daño sobre la imagen final (sea recorte o fallback)
        ratio, severidad_label, mask_disease = expert_system.sistema_experto_severidad(roi_final)
        
        # --- PASO 4: PUENTE NUMPY -> PIL -> TENSOR (Para PyTorch) ---
        pil_image = Image.fromarray(roi_final)
        img_tensor = self.transforms(pil_image).unsqueeze(0).to(self.device)

        # --- PASO 5: INFERENCIA HÍBRIDA ---
        with torch.no_grad():
            features_1280 = self.vision_model(img_tensor).cpu().numpy()
            
        # Pipeline Matemático
        features_scaled = self.scaler.transform(features_1280)
        features_lda = self.lda.transform(features_scaled)
        
        # Fuzzy Prediction
        u, _, _, _, _, _ = fuzz.cluster.cmeans_predict(
            features_lda.T, self.centers, m=self.m, error=0.005, maxiter=1000
        )
        
        # Ordenar resultados
        probs = self._reorder_probabilities(u.flatten())
        
        # --- PASO 6: FORMATEO DE RESULTADOS ---
        labels = ['Black Rot', 'Esca', 'Healthy', 'Leaf Blight']
        
        result = {label: round(float(prob) * 100, 2) for label, prob in zip(labels, probs)}
        winner = labels[np.argmax(probs)]
        confidence = float(np.max(probs) * 100)
        
        # Generar imagen base64 de la máscara de enfermedad
        mask_base64 = None
        if mask_disease is not None:
            _, buffer = cv2.imencode('.jpg', mask_disease)
            mask_base64 = base64.b64encode(buffer).decode('utf-8')

        # JSON FINAL ESTRUCTURADO
        return {
            "status": "success",
            "diagnosis": {
                "disease": winner,
                "confidence": round(confidence, 2),
                "fuzzy_probs": result
            },
            "expert_analysis": {
                "severity_label": severidad_label,
                "tissue_damage_percent": round(ratio, 2),
                "roi_detected": roi_detected  # Avisamos al frontend si encontramos la hoja o no
            },
            "visuals": {
                "damage_mask_base64": mask_base64
            }
        }