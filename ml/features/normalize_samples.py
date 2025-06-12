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


def normalize_samples(root_path):
    """
    Normaliza todas las muestras dentro de un directorio de palabra.

    Para cada subcarpeta encontrada en `root_path`, lee los frames de la muestra,
    los ajusta a una cantidad fija mediante interpolación o recorte, y sobrescribe
    los archivos con los frames normalizados.

    Args:
        root_path (str): Ruta a la carpeta que contiene las subcarpetas con muestras.

    Returns:
        None: Esta función no retorna ningún valor. Modifica los archivos en disco.
    """

    print(f"🧪 Entrando a normalize_samples con path: {root_path}")

    sample_folders = [
        name
        for name in os.listdir(root_path)
        if os.path.isdir(os.path.join(root_path, name))
    ]

    print(f"📁 Se encontraron {len(sample_folders)} carpetas de muestra")
    print("👉 Carpetas:", sample_folders)

    print(f"🔄 Normalizando {len(sample_folders)} muestras...")

    for i, folder in enumerate(sample_folders, start=1):
        sample_path = os.path.join(root_path, folder)
        frames = read_frames_from_directory(sample_path)
        print(f"📷 Leyendo muestra: {sample_path} - {len(frames)} frames")

        if not frames:
            print(f"⚠️  Muestra vacía omitida: {folder}")
            print(f"⚠️ Muestra vacía omitida: {sample_path}")
            continue

        normalized = normalize_frames(frames)
        clear_directory(sample_path)
        save_normalized_frames(sample_path, normalized)

        print(f"✔️  Muestra {i}/{len(sample_folders)} normalizada: {folder}")
