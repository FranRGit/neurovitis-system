import cv2
import numpy as np 
def sistema_experto_severidad(imagen_roi_rgb):
    """Calcula el porcentaje de daño tisular."""
    if imagen_roi_rgb is None:
        return 0.0, "Error de Procesamiento", None

    # HSV para detectar verde
    hsv = cv2.cvtColor(imagen_roi_rgb, cv2.COLOR_RGB2HSV)
    lower_green = np.array([35, 30, 30])
    upper_green = np.array([85, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # Máscara de la hoja (Otsu)
    gray = cv2.cvtColor(imagen_roi_rgb, cv2.COLOR_RGB2GRAY)
    _, mask_leaf = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Limpieza
    kernel = np.ones((5,5), np.uint8)
    mask_leaf = cv2.morphologyEx(mask_leaf, cv2.MORPH_CLOSE, kernel)

    # Cálculo: (Hoja) - (Verde) = Enfermedad
    leaf_green_parts = cv2.bitwise_and(mask_leaf, mask_green)
    mask_disease = cv2.subtract(mask_leaf, leaf_green_parts)

    pixels_total = cv2.countNonZero(mask_leaf)
    pixels_disease = cv2.countNonZero(mask_disease)

    if pixels_total == 0: 
        ratio = 0.0
    else: 
        ratio = (pixels_disease / pixels_total) * 100

    if ratio < 5.0: diagnostico = "Leve / Incierto"
    elif ratio < 25.0: diagnostico = "Daño LEVE"
    elif ratio < 50.0: diagnostico = "Daño MODERADO"
    else: diagnostico = "Daño SEVERO"

    return ratio, diagnostico, mask_disease