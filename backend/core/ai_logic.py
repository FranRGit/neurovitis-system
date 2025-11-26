import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import joblib
import skfuzzy as fuzz
import os
from PIL import Image
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler

# --- 1. MODELO VISUAL ---
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
        
        print(">>> Cargando MobileNetV2...")
        self.vision_model = FeatureExtractor().to(self.device)
        
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(">>> Cargando (LDA + Fuzzy)...")
        self._load_math_models()
        
    def _load_math_models(self):
        # Cargar Scaler y LDA
        self.scaler = joblib.load(os.path.join(self.models_dir, 'modelo_scaler.pkl'))
        self.lda = joblib.load(os.path.join(self.models_dir, 'modelo_lda.pkl'))
        
        # Cargar Fuzzy
        self.centers = np.load(os.path.join(self.models_dir, 'fcm_centers.npy'))
        self.mapping = joblib.load(os.path.join(self.models_dir, 'fcm_mapping.pkl'))
        
        # Configuración Fuzzy
        self.n_clusters = self.centers.shape[0]
        self.m = 1.2 # El valor que nos funcionó con LDA

    def _reorder_probabilities(self, u_raw):
        u_ordered = np.zeros(self.n_clusters)
        for cluster_id, real_label_id in self.mapping.items():
            u_ordered[real_label_id] += u_raw[cluster_id]
        return u_ordered

    def predict(self, image_file):
        # 1. Pre-procesar Imagen
        img = Image.open(image_file).convert('RGB')
        img_tensor = self.transforms(img).unsqueeze(0).to(self.device)
        
        # 2. Extracción con CNN
        with torch.no_grad():
            features_1280 = self.vision_model(img_tensor).cpu().numpy()
            
        # 3. Reducción Scaler + LDA
        features_scaled = self.scaler.transform(features_1280)
        features_lda = self.lda.transform(features_scaled)
        
        # 4. Razonamiento 
        # Transponemos features_lda para skfuzzy: (Features, 1)
        u, _, _, _, _, _ = fuzz.cluster.cmeans_predict(
            features_lda.T, self.centers, m=self.m, error=0.005, maxiter=1000
        )
        
        # 5. Ordenar y Formatear
        probs = self._reorder_probabilities(u.flatten())
        
        # Etiquetas oficiales (en orden 0, 1, 2, 3)
        labels = ['Black Rot', 'Esca', 'Healthy', 'Leaf Blight']
        
        result = {label: round(float(prob) * 100, 2) for label, prob in zip(labels, probs)}
        winner = labels[np.argmax(probs)]
        confidence = float(np.max(probs) * 100)
        
        return {
            "diagnosis": winner,
            "confidence": confidence,
            "details": result
        }