import cv2
import numpy as np

def cargar_imagen_desde_memoria(file_stream):
    """Lee los bytes de Flask y devuelve una imagen RGB limpia."""
    # Leer el stream a bytes
    file_bytes = np.frombuffer(file_stream.read(), np.uint8)
    # Decodificar
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img_bgr is None:
        return None
    
    # IMPORTANTE: Trabajamos todo en RGB internamente
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def normalizar_hoja(img_rgb):
    """Aplica Dynamic ROI y CLAHE. Retorna la imagen procesada lista para AI."""
    try:
        # 1. Escala de grises y Otsu (Auto-umbral)
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 2. Contornos
        contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contornos:
            return None # Falló detección

        # Tomar el contorno más grande
        c = max(contornos, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)

        # 3. Recorte (ROI) con padding
        pad = 20
        h_img, w_img, _ = img_rgb.shape
        y1, y2 = max(0, y-pad), min(h_img, y+h+pad)
        x1, x2 = max(0, x-pad), min(w_img, x+w+pad)
        
        roi_rgb = img_rgb[y1:y2, x1:x2]

        # 4. CLAHE (Mejora de iluminación)
        lab = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l_blurred = cv2.GaussianBlur(l, (3, 3), 0) # Blur para quitar ruido
        clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8,8))
        cl = clahe.apply(l_blurred)
        limg = cv2.merge((cl, a, b))
        
        roi_clahe_rgb = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

        return roi_clahe_rgb
    except Exception as e:
        print(f"Error en normalización: {e}")
        return None