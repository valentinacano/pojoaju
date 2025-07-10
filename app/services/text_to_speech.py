"""
Conversión de texto a voz con pyttsx3.

Este módulo proporciona una función que utiliza el motor de texto a voz (TTS) local
`pyttsx3` para reproducir texto en voz alta. Es útil para aplicaciones accesibles
que no dependen de conexión a internet.
"""

import pyttsx3


def text_to_speech(text):
    """
    Convierte texto a voz utilizando la librería pyttsx3.

    Usa un motor TTS local para reproducir el texto recibido. No requiere conexión a internet
    y funciona en sistemas operativos compatibles con los motores de voz del sistema.

    Args:
        text (str): Texto que se desea reproducir en voz.

    Returns:
        None: Esta función no retorna ningún valor. Solo ejecuta la reproducción de audio.
    """
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
