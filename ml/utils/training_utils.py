from app.database.database_utils import fetch_keypoints_by_words
from ml.utils.keypoints_utils import group_keypoints_by_word_and_sample


def get_sequences_and_labels(word_ids):
    """
    Recupera todas las secuencias y etiquetas de la base de datos.

    Combina el fetch de los word_ids, la consulta de keypoints y el agrupamiento.

    Returns:
        tuple[list[list], list[int]]: Lista de secuencias de keypoints y sus etiquetas.
    """
    keypoints_data = fetch_keypoints_by_words(word_ids)
    return group_keypoints_by_word_and_sample(keypoints_data, word_ids)
