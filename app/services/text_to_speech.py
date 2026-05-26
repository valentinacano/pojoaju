"""
Texto a voz — única implementación del proyecto.

Usa el comando 'say' nativo de macOS para reproducir texto en voz alta.
Es más confiable que pyttsx3 en macOS porque no tiene problemas con hilos.

En Linux/Windows usa pyttsx3 como fallback.
"""

import sys
import threading
import subprocess


def text_to_speech(text: str):
    """
    Reproduce un texto en voz alta.

    En macOS usa el comando 'say' del sistema.
    En otros sistemas usa pyttsx3.

    Args:
        text: texto a pronunciar.
    """
    try:
        if sys.platform == "darwin":
            # macOS — comando nativo, siempre funciona
            subprocess.run(["say", "-v", "Paulina", text], check=False)
        else:
            import pyttsx3

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
