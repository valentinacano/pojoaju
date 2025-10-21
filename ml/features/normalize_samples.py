"""
Normalización de muestras desde un directorio de palabra.

Este módulo forma parte del pipeline de preprocesamiento. Recorre todas las subcarpetas 
que representan muestras individuales dentro de una carpeta de palabra, lee los frames 
de cada muestra, los normaliza a una cantidad fija mediante interpolación o recorte, 
y sobrescribe los archivos originales con los frames procesados.

Está diseñado para asegurar que todas las muestras tengan la misma longitud, 
facilitando su uso en modelos de entrenamiento secuencial.

Estructura esperada:
- `root_path/` → contiene carpetas `sample_YYYYMMDD.../` con imágenes `.jpg` secuenciales.
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

    Para cada subcarpeta encontrada en `root_path`, realiza los siguientes pasos:
    1. Lee todos los frames de la muestra.
    2. Aplica interpolación o recorte para que todos los sets de frames tengan igual longitud.
    3. Limpia el contenido de la carpeta original.
    4. Guarda los frames normalizados con nombres secuenciales.

    Este proceso es esencial para preparar los datos de entrada a modelos de tipo LSTM
    o cualquier arquitectura que requiera secuencias homogéneas en longitud.

    Args:
        root_path (str): Ruta a la carpeta que contiene subcarpetas con muestras (una por secuencia).

    Returns:
        None: Esta función modifica directamente los archivos en disco, sobrescribiendo los originales.
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
            print(f"⚠️ Muestra vacía omitida: {folder}")
            continue

        normalized = normalize_frames(frames)
        clear_directory(sample_path)
        save_normalized_frames(sample_path, normalized)

        print(f"✔️ Muestra {i}/{len(sample_folders)} normalizada: {folder}")
