"""
Texto a voz — única implementación del proyecto.

Usa pyttsx3 (local, sin internet). Se ejecuta en un hilo separado
para no bloquear la interfaz web ni la cámara.
"""

import threading
import pyttsx3


def text_to_speech(text: str):
    """
    Reproduce un texto en voz alta de forma sincrónica.

    Args:
        text: texto a pronunciar.
    """
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"⚠️ TTS error: {e}")


def text_to_speech_async(text: str):
    """
    Reproduce un texto en voz alta en un hilo separado (no bloquea).

    Args:
        text: texto a pronunciar.
    """
    threading.Thread(target=text_to_speech, args=(text,), daemon=True).start()