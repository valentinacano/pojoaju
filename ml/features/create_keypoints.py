"""
Generación de keypoints para una palabra a partir de secuencias de imágenes.

Este módulo procesa una carpeta de muestras (frames) correspondientes a una palabra
específica, extrae los keypoints utilizando MediaPipe Holistic y guarda el resultado
en un archivo HDF5 para su uso posterior en modelos de machine learning.
"""

import os
import pandas as pd
from mediapipe.python.solutions.holistic import Holistic
from ml.utils.create_keypoints import get_keypoints, insert_keypoints_sequence


def create_keypoints(word_id, words_path, hdf_path):
    """
    Crea y guarda los keypoints de todas las muestras asociadas a una palabra.

    Procesa todas las carpetas de muestras (una por ejemplo capturada con `capture_samples`),
    extrae los keypoints usando MediaPipe Holistic y guarda el resultado en un archivo `.h5`.

    Args:
        word_id (str): Nombre o identificador de la palabra (coincide con el nombre de la carpeta).
        words_path (str): Ruta a la carpeta raíz que contiene todas las palabras y sus muestras.
        hdf_path (str): Ruta al archivo `.h5` donde se guardarán los keypoints generados.

    Returns:
        None: Esta función no retorna ningún valor. Guarda los datos en el archivo `.h5`.
    """
    data = pd.DataFrame([])
    frames_path = os.path.join(words_path, word_id)

    with Holistic() as holistic:
        print(f'Creando keypoints de "{word_id}"...')
        sample_list = os.listdir(frames_path)
        sample_count = len(sample_list)

        for n_sample, sample_name in enumerate(sample_list, start=1):
            sample_path = os.path.join(frames_path, sample_name)
            keypoints_sequence = get_keypoints(holistic, sample_path)
            data = insert_keypoints_sequence(data, n_sample, keypoints_sequence)
            print(f"{n_sample}/{sample_count}", end="\r")

    data.to_hdf(hdf_path, key="data", mode="w")
    print(f"Keypoints creados! ({sample_count} muestras)", end="\n")
