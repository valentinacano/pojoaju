"""
Traducción usando Gemini:

1. translate_signs_to_spanish(): señas LSPy → español natural
2. spanish_to_lspy_sequence(): español → secuencia de palabras del diccionario
"""

import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ---------------------------------------------------------------------------
# Prompt 1: señas → español
# ---------------------------------------------------------------------------

SIGNS_TO_SPANISH_PROMPT = """Sos un experto en Lengua de Señas Paraguaya (LSPy).

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
# Prompt 2: español → secuencia LSPy del diccionario
# ---------------------------------------------------------------------------

SPANISH_TO_LSPY_PROMPT = """Sos un experto en Lengua de Señas Paraguaya (LSPy).

Tu tarea es convertir una frase en español a una secuencia de señas usando ÚNICAMENTE las palabras del diccionario disponible.

Reglas:
- Usá SOLO las palabras del diccionario disponible (te las voy a pasar)
- Eliminá artículos (el, la, los, las, un, una) si no están en el diccionario
- Eliminá verbos copulativos (ser, estar) si no están en el diccionario
- Reordenás según la gramática LSPy: TIEMPO + SUJETO + ACCIÓN o TEMA + INFORMACIÓN
- Si una palabra de la frase no tiene equivalente en el diccionario, omitila
- Si ninguna palabra de la frase está en el diccionario, devolvé una lista vacía

Respondé ÚNICAMENTE con un array JSON de strings, sin explicaciones ni texto extra.

Ejemplos:
- Frase: "Mi mamá es muy linda", Diccionario: ["mamá", "lindo", "mucho", "hola"]
  → ["mamá", "lindo", "mucho"]
- Frase: "Hola, ¿cómo estás?", Diccionario: ["hola", "bien", "mal"]
  → ["hola"]
- Frase: "Buenos días", Diccionario: ["día", "hola", "mamá"]
  → ["día"]"""


# ---------------------------------------------------------------------------
# Funciones
# ---------------------------------------------------------------------------


def _get_model():
    if not GEMINI_API_KEY:
        return None
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel("gemini-2.5-flash-lite")


def translate_signs_to_spanish(signs: list[str]) -> str | None:
    """
    Traduce una secuencia de señas al español natural usando Gemini.

    Args:
        signs: lista de palabras detectadas por el modelo LSTM.

    Returns:
        Frase en español natural, o las señas sin traducir si hay error.
    """
    if not signs:
        return None

    model = _get_model()
    if not model:
        print("⚠️ GEMINI_API_KEY no configurada en .env")
        return " ".join(signs)

    try:
        sequence = " ".join([s.upper() for s in signs])
        prompt = f"{SIGNS_TO_SPANISH_PROMPT}\n\nSecuencia de señas: {sequence}"
        response = model.generate_content(prompt)
        translation = response.text.strip()
        print(f"🔤 Señas: {sequence}")
        print(f"📝 Traducción: {translation}")
        return translation
    except Exception as e:
        print(f"⚠️ Error al traducir con Gemini: {e}")
        return " ".join(signs)


def spanish_to_lspy_sequence(phrase: str, available_words: list[str]) -> list[str]:
    """
    Convierte una frase en español a una secuencia de palabras del diccionario LSPy.

    Gemini elige qué palabras del diccionario usar y en qué orden,
    respetando la gramática de la LSPy.

    Args:
        phrase: frase en español (ej. "Mi mamá es muy linda")
        available_words: palabras disponibles en el diccionario (ej. ["mamá", "hola", "día"])

    Returns:
        Lista de palabras del diccionario en orden LSPy, o lista vacía si no hay match.
    """
    if not phrase or not available_words:
        return []

    model = _get_model()
    if not model:
        print("⚠️ GEMINI_API_KEY no configurada en .env")
        # Fallback: buscar palabras de la frase que estén en el diccionario
        words = phrase.lower().split()
        return [w for w in words if w in available_words]

    try:
        dictionary_str = ", ".join(available_words)
        prompt = (
            f"{SPANISH_TO_LSPY_PROMPT}\n\n"
            f"Diccionario disponible: [{dictionary_str}]\n"
            f'Frase en español: "{phrase}"\n'
            f"Respondé solo con el array JSON:"
        )
        response = model.generate_content(prompt)
        text = response.text.strip()

        # Limpiar posibles backticks de markdown
        text = text.replace("```json", "").replace("```", "").strip()
        print(f"🤖 Respuesta cruda de Gemini: {text}")
        sequence = json.loads(text)

        # Validar que todas las palabras estén en el diccionario
        valid = [
            w for w in sequence if w.lower() in [a.lower() for a in available_words]
        ]

        print(f"📝 Frase: {phrase}")
        print(f"🤟 Secuencia LSPy: {valid}")
        return valid

    except Exception as e:
        print(f"⚠️ Error al convertir frase a LSPy: {e}")
        # Fallback simple
        words = phrase.lower().split()
        return [w for w in words if w in [a.lower() for a in available_words]]
