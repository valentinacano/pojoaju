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


def normalize_samples(root_path, target_frame_count=15):
    """
    Normaliza todas las muestras dentro de un directorio de palabra.

    Para cada subcarpeta encontrada en `root_path`, lee los frames de la muestra,
    los ajusta a una cantidad fija mediante interpolación o recorte, y sobrescribe
    los archivos con los frames normalizados.

    Args:
        root_path (str): Ruta a la carpeta que contiene las subcarpetas con muestras.
        target_frame_count (int): Cantidad fija de frames a la que se deben normalizar las muestras.

    Returns:
        None: Esta función no retorna ningún valor. Modifica los archivos en disco.
    """
    sample_folders = [
        name
        for name in os.listdir(root_path)
        if os.path.isdir(os.path.join(root_path, name))
    ]

    print(f"🔄 Normalizando {len(sample_folders)} muestras...")

    for i, folder in enumerate(sample_folders, start=1):
        sample_path = os.path.join(root_path, folder)
        frames = read_frames_from_directory(sample_path)

        if not frames:
            print(f"⚠️  Muestra vacía omitida: {folder}")
            continue

        normalized = normalize_frames(frames, target_frame_count)
        clear_directory(sample_path)
        save_normalized_frames(sample_path, normalized)

        print(f"✔️  Muestra {i}/{len(sample_folders)} normalizada: {folder}")
