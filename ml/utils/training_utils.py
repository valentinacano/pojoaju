"""
Utilidades de entrenamiento para modelos de reconocimiento de lenguaje de señas.

Este módulo facilita la preparación de los datos de entrenamiento. Recupera los keypoints
desde la base de datos, los agrupa por muestra y los convierte en secuencias listas para
entrenar modelos de clasificación.
"""

from app.database.database_utils import fetch_keypoints_by_words
from ml.utils.keypoints_utils import group_keypoints_by_word_and_sample


def get_sequences_and_labels(word_ids):
    """
    Recupera todas las secuencias de keypoints y sus etiquetas desde la base de datos.

    Esta función realiza tres pasos principales:
    1. Consulta los vectores de keypoints asociados a los `word_ids` especificados.
    2. Agrupa los keypoints por palabra y muestra.
    3. Devuelve las secuencias preparadas para entrenamiento y las etiquetas correspondientes.

    Args:
        word_ids (list[int]): Lista de identificadores de palabras cuyos keypoints se desean recuperar.

    Returns:
        tuple[list[list], list[int]]:
            - Lista de secuencias de keypoints (una por muestra).
            - Lista de etiquetas numéricas correspondientes a cada secuencia.
    """
    keypoints_data = fetch_keypoints_by_words(word_ids)
    return group_keypoints_by_word_and_sample(keypoints_data, word_ids)
