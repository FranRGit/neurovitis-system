# backend/core/agent_logic.py
import os
import google.generativeai as genai
from dotenv import load_dotenv
from pathlib import Path
from core import db_logic

# Cargar entorno
load_dotenv()

# Configurar Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    print("ADVERTENCIA: No hay GOOGLE_API_KEY. El agente no funcionará.")


def formatear_fuzzy_probs(fuzzy_dict):
    if not fuzzy_dict: return "No disponible"
    # Ordenar de mayor a menor probabilidad
    sorted_probs = sorted(fuzzy_dict.items(), key=lambda x: x[1], reverse=True)
    return ", ".join([f"{k}: {v}%" for k, v in sorted_probs])

def neurobot_agente_mensaje(mensaje_usuario, datos_contexto):
    if not GOOGLE_API_KEY:
        return "Error: Cerebro IA no configurado."

    # --- LÓGICA DE DETECCIÓN DE INTENCIÓN (ROUTER) ---
    msg_upper = mensaje_usuario.upper()
    
    # CASO A: EL USUARIO PIDE UN RESUMEN HISTÓRICO
    if "RESUMEN" in msg_upper or "REPORTE" in msg_upper or "HISTORIAL" in msg_upper:
        return generar_resumen_historico(mensaje_usuario)
    
    # CASO B: ANÁLISIS DE LA IMAGEN ACTUAL (Contexto Frontend)
    return analisis_imagen_actual(mensaje_usuario, datos_contexto)

def generar_resumen_historico(mensaje_usuario):
    historial = db_logic.obtener_resumen_historico(20) # Últimos 20 casos
    
    if not historial:
        return "No tengo suficientes datos históricos en la base de datos para generar un resumen."

    # Convertir datos a texto para el prompt
    data_str = "\n".join([f"- {h['created_at'][:10]}: {h['disease']} ({h['confidence']}%), Severidad: {h['severity_label']}" for h in historial])

    system_prompt = f"""
    ACTÚA COMO: Jefe de Sanidad Agrícola.
    TAREA: Analizar el historial reciente de escaneos y dar conclusiones.
    
    DATOS DE LOS ÚLTIMOS ESCANEOS:
    {data_str}
    
    INSTRUCCIONES DE RESPUESTA:
    1. Usa texto plano con secciones claras (TÍTULOS EN MAYÚSCULAS).
    2. Identifica cuál es la enfermedad más frecuente.
    3. Si ves muchos casos de 'Healthy', felicita. Si ves plagas recurrentes, sugiere un plan de acción general.
    4. NO listes los registros uno por uno. Haz una síntesis.
    """
    
    return llamar_gemini(system_prompt)

def analisis_imagen_actual(mensaje_usuario, datos_contexto):
    """Sub-agente encargado del diagnóstico puntual con Lógica Difusa"""
    
    # 1. Preparar datos ricos
    if datos_contexto and 'disease' in datos_contexto:
        fuzzy_text = formatear_fuzzy_probs(datos_contexto.get('fuzzy_probs', {}))
        
        contexto_str = f"""
        DATOS DEL ALGORITMO HÍBRIDO (NEURO-FUZZY):
        - Clase Ganadora: {datos_contexto.get('disease')} (Confianza: {datos_contexto.get('confidence')}%)
        - Distribución Difusa Completa: [{fuzzy_text}]
        
        DATOS DEL SISTEMA EXPERTO (VISIÓN ARTIFICIAL):
        - Nivel de Severidad: {datos_contexto.get('severity')}
        - Daño Tisular Detectado: {datos_contexto.get('damage_percent')}%
        - ¿Se detectó hoja correctamente?: {'Sí' if datos_contexto.get('roi_detected') else 'No (Se analizó imagen completa)'}
        """
    else:
        contexto_str = "El usuario no está viendo ningún análisis activo en pantalla."

    # 2. PROMPT DE INGENIERÍA (Chain of Thought)
    system_prompt = f"""
    ERES: 'NeuroBot', un Ingeniero Agrónomo experto.
    
    TU MISIÓN: Interpretar los datos matemáticos del sistema NeuroVitis y traducirlos a consejos prácticos.
    
    DATOS TÉCNICOS DE LA IMAGEN ACTUAL:
    {contexto_str}
    
    REGLAS DE RAZONAMIENTO CRÍTICO (IMPORTANTE):
    1. ANÁLISIS DE INCERTIDUMBRE: 
        - Mira la "Distribución Difusa". 
        - Si la clase ganadora es "Healthy" pero hay más de un 30% de probabilidad en alguna enfermedad (ej. Black Rot), ADVIERTE AL USUARIO. Dile: "Aunque el sistema clasifica como Sana, veo indicios preocupantes de X".
        - Si la confianza es baja (<60%), sugiere tomar otra foto con mejor luz.
        
    2. ANÁLISIS DE SEVERIDAD:
        - Si el daño tisular es >20%, el tono debe ser de URGENCIA.
        - Si el daño es <5%, sugiere monitoreo preventivo.

    3. FORMATO DE RESPUESTA (Usa este esquema):
        ANÁLISIS TÉCNICO
        (Tu interpretación de los datos y la incertidumbre)
        
        RECOMENDACIÓN
        (Fungicidas, poda, riego, etc.)
        
        **ALERTA** (Opcional)
        (Solo si hay ambigüedad fuzzy o severidad alta)

    PREGUNTA DEL USUARIO:
    "{mensaje_usuario}"
    """
    
    return llamar_gemini(system_prompt)

def llamar_gemini(prompt):
    try:
        # Usamos el modelo flash por velocidad y cuota
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error de conexión con IA: {str(e)}"