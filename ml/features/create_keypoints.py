"""
Generación de keypoints para una palabra a partir de secuencias de imágenes.

Este módulo procesa carpetas de muestras (frames) correspondientes a una palabra específica,
extrae los keypoints utilizando MediaPipe Holistic y los guarda directamente en una base
de datos PostgreSQL, utilizando la tabla `keypoints`.
"""

import os
from mediapipe.python.solutions.holistic import Holistic
from ml.utils.keypoints_utils import get_keypoints
from app.database.database_utils import save_keypoints_to_db


def create_keypoints(word_name, words_path, word_id):
    """
        Extrae keypoints desde secuencias de imágenes y los guarda en la base de datos.

        Recorre todas las subcarpetas (muestras) asociadas a una palabra, extrae los keypoints
    de cada frame usando MediaPipe Holistic, y guarda los resultados en la tabla `keypoints`.

        Args:
            word_name (str): Nombre descriptivo de la palabra (usado para impresión en consola).
            words_path (str): Ruta raíz que contiene carpetas por palabra, cada una con subcarpetas de muestras.
            word_id (int): Identificador numérico de la palabra en la base de datos (columna `word_id`).

        Returns:
            None: Esta función no retorna ningún valor. Inserta los datos directamente en la base de datos.
    """
    word_path = os.path.join(words_path, word_name)

    sample_folders = [
        name
        for name in os.listdir(word_path)
        if os.path.isdir(os.path.join(word_path, name))
    ]

    print(f"🧠 Procesando palabra '{word_name}' (ID: {word_id})")

    with Holistic() as model:
        for i, folder in enumerate(sample_folders, start=1):
            sample_path = os.path.join(word_path, folder)
            keypoints_seq = get_keypoints(model, sample_path)
            save_keypoints_to_db(word_id, i, keypoints_seq)
