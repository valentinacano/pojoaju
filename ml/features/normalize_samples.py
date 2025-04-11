"""
Normalización de muestras completas desde un directorio de palabra.

Este módulo permite recorrer todas las muestras (carpetas) dentro de una palabra,
leer sus frames, normalizar la longitud a una cantidad fija, y guardar los resultados
en la misma carpeta.
"""

import os

from ml.utils.normalize_samples import (
    read_frames_from_directory,
    clear_directory,
    save_normalized_frames,
    normalize_frames,
)


def normalize_samples(word_directory, target_frame_count=15):
    """
    Normaliza todas las muestras dentro de un directorio de palabra.

    Para cada subcarpeta encontrada en `word_directory`, lee los frames, los normaliza
    a una cantidad fija (`target_frame_count`), elimina los archivos originales y guarda
    los nuevos frames normalizados.

    Args:
        word_directory (str): Ruta a la carpeta que contiene las muestras (una por subcarpeta).
        target_frame_count (int): Cantidad fija de frames a la que se debe normalizar cada muestra.

    Returns:
        None: Esta función no retorna ningún valor. Sobrescribe las muestras en disco.
    """
    for sample_name in os.listdir(word_directory):
        sample_directory = os.path.join(word_directory, sample_name)
        if os.path.isdir(sample_directory):
            frames = read_frames_from_directory(sample_directory)
            normalized_frames = normalize_frames(frames, target_frame_count)
            clear_directory(sample_directory)
            save_normalized_frames(sample_directory, normalized_frames)
