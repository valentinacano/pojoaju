"""
Generación de keypoints para una palabra a partir de secuencias de imágenes.

Este módulo procesa carpetas de muestras (frames) de una palabra específica,
extrae los keypoints utilizando MediaPipe Holistic y guarda el resultado en un
archivo HDF5 compatible con modelos de machine learning.
"""

import os
import pandas as pd
from mediapipe.python.solutions.holistic import Holistic
from ml.utils.keypoints_utils import get_keypoints, insert_keypoints_sequence


def create_keypoints(word_name, words_path, hdf_path):
    """
    Crea y guarda los keypoints de todas las muestras asociadas a una palabra.

    Recorre todas las subcarpetas dentro de una palabra, interpreta cada una como una muestra.
    Usa MediaPipe Holistic para extraer los vectores de keypoints por frame, y guarda la
    información en un archivo HDF5.

    Args:
        word_name (str): Nombre de la palabra (coincide con el nombre de la carpeta).
        words_path (str): Ruta raíz que contiene todas las palabras y sus carpetas de muestras.
        hdf_path (str): Ruta al archivo `.h5` donde se almacenarán los vectores resultantes.

    Returns:
        None: Esta función no retorna ningún valor. Guarda el archivo `.h5` en disco.
    """
    data = pd.DataFrame([])
    word_path = os.path.join(words_path, word_name)

    sample_folders = [
        name
        for name in os.listdir(word_path)
        if os.path.isdir(os.path.join(word_path, name))
    ]

    print(f"🧠 Creando keypoints para '{word_name}'...")

    with Holistic() as model:
        for i, folder in enumerate(sample_folders, start=1):
            sample_path = os.path.join(word_path, folder)
            keypoints_seq = get_keypoints(model, sample_path)
            data = insert_keypoints_sequence(data, i, keypoints_seq)
            print(f"✔️  Muestra {i}/{len(sample_folders)} procesada")

    data.to_hdf(hdf_path, key="data", mode="w")
    print(f"\n✅ Keypoints guardados en: {hdf_path}")
