"""
Traducción de secuencias de señas (LSPy) a español natural usando Gemini.

Toma una lista de palabras detectadas por el modelo LSTM y las traduce
a una frase en español gramaticalmente correcta, respetando las reglas
de la Lengua de Señas Paraguaya.
"""

import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

SYSTEM_PROMPT = """Sos un experto en Lengua de Señas Paraguaya (LSPy).

Tu tarea es traducir secuencias de señas detectadas por una cámara al español natural escrito.

Reglas de la LSPy que debés aplicar:
- Los artículos (el, la, los, las, un, una) generalmente se omiten en LSPy pero deben agregarse en español
- Los verbos copulativos (ser, estar) se omiten en LSPy pero deben agregarse cuando corresponda
- La posesión va después del sustantivo: MAMÁ MÍA = "mi mamá"
- El orden puede ser TEMA + INFORMACIÓN o SUJETO + ACCIÓN
- El tiempo va primero: AYER CASA IR = "ayer fui a casa"
- Los pronombres a veces se omiten

Ejemplos:
- MAMÁ MÍA LINDA MUCHO → "Mi mamá es muy linda"
- AYER CASA MÍA IR → "Ayer fui a mi casa"
- PAPÁ MÍO TRABAJAR MUCHO → "Mi papá trabaja mucho"
- HOLA → "Hola"
- GRACIAS → "Gracias"

Respondé SOLO con la traducción al español, sin explicaciones ni comillas."""


# ---------------------------------------------------------------------------
# Función principal
# ---------------------------------------------------------------------------


def translate_signs_to_spanish(signs: list[str]) -> str | None:
    """
    Traduce una secuencia de señas al español natural usando Gemini.

    Args:
        signs: lista de palabras detectadas por el modelo LSTM.
               Ejemplo: ["MAMÁ", "MÍA", "LINDA", "MUCHO"]

    Returns:
        Frase en español natural, o None si hay un error.
    """
    if not signs:
        return None

    if not GEMINI_API_KEY:
        print("⚠️ GEMINI_API_KEY no configurada en .env")
        return " ".join(signs)  # fallback: mostrar las señas sin traducir

    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-2.5-flash-lite")

        sequence = " ".join([s.upper() for s in signs])
        prompt = f"{SYSTEM_PROMPT}\n\nSecuencia de señas: {sequence}"

        response = model.generate_content(prompt)
        translation = response.text.strip()

        print(f"🔤 Señas: {sequence}")
        print(f"📝 Traducción: {translation}")

        return translation

    except Exception as e:
        print(f"⚠️ Error al traducir con Gemini: {e}")
        return " ".join(signs)  # fallback: mostrar las señas sin traducir
