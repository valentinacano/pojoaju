"""
Generación de keypoints a partir de secuencias de imágenes para una palabra.

Este módulo forma parte del pipeline de preprocesamiento y extracción de datos. 
Procesa carpetas de muestras asociadas a una palabra específica, extrae los keypoints 
utilizando MediaPipe Holistic y guarda los vectores generados directamente en la base 
de datos PostgreSQL, utilizando la tabla `keypoints`.

Se utiliza como etapa final después de la captura y normalización de muestras.
"""


import os
from mediapipe.python.solutions.holistic import Holistic
from ml.utils.keypoints_utils import get_keypoints
from app.database.database_utils import insert_keypoints


def create_keypoints(word_name, words_path, word_id):
    """
    Extrae keypoints desde secuencias de imágenes y los guarda en la base de datos.

    Para cada subcarpeta encontrada dentro de la carpeta de palabra, esta función
    recorre los frames, extrae los vectores de keypoints mediante MediaPipe Holistic,
    y los inserta en la tabla `keypoints` de PostgreSQL.

    Args:
        word_name (str): Nombre descriptivo de la palabra (usado para impresión en consola).
        words_path (str): Ruta raíz que contiene carpetas por palabra, cada una con subcarpetas de muestras.
        word_id (int | bytes): Identificador único de la palabra en la base de datos (columna `word_id`).

    Returns:
        None: Esta función no retorna ningún valor. Inserta los vectores directamente en la base de datos.
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
            insert_keypoints(word_id, i, keypoints_seq)
