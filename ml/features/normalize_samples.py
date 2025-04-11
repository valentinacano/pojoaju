"""
Normalización de muestras desde un directorio de palabra.

Este módulo recorre todas las carpetas de muestras dentro de una palabra, 
lee los frames, los normaliza a una cantidad fija y guarda los resultados 
en la misma carpeta, sobrescribiendo los archivos originales.
"""

import os
from ml.utils.normalize_utils import (
    read_frames_from_directory,
    clear_directory,
    save_normalized_frames,
    normalize_frames,
)


def normalize_samples(word_directory, target_frame_count=15):
    """
    Normaliza todas las muestras dentro de un directorio de palabra.

    Para cada subcarpeta encontrada en `word_directory`, lee los frames,
    los normaliza a una cantidad fija (`target_frame_count`), elimina los
    archivos originales y guarda los nuevos frames normalizados.

    Args:
        word_directory (str): Ruta a la carpeta que contiene las muestras.
        target_frame_count (int): Cantidad fija de frames por muestra.

    Returns:
        None
    """
    sample_folders = [
        name
        for name in os.listdir(word_directory)
        if os.path.isdir(os.path.join(word_directory, name))
    ]

    print(f"🔄 Normalizando {len(sample_folders)} muestras...")

    for i, folder in enumerate(sample_folders, start=1):
        sample_path = os.path.join(word_directory, folder)
        frames = read_frames_from_directory(sample_path)

        if not frames:
            print(f"⚠️  Muestra vacía omitida: {folder}")
            continue

        normalized = normalize_frames(frames, target_frame_count)
        clear_directory(sample_path)
        save_normalized_frames(sample_path, normalized)

        print(f"✔️  Muestra {i}/{len(sample_folders)} normalizada: {folder}")
