import os, cv2
import numpy as np
import pandas as pd

from mediapipe.python.solutions.holistic import Holistic
from ml.utils.keypoints_utils import (
    extract_keypoints,
    get_keypoints,
    insert_keypoints_sequence,
    group_keypoints_by_word_and_sample,
)


def _crear_frame_dummy():
    """
    Crea una imagen aleatoria (no negra) simulando un frame con contenido.

    Returns:
        np.ndarray: Imagen RGB con datos aleatorios.
    """
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


def test_extract_keypoints_completo():
    """
    Verifica que `extract_keypoints` devuelva un vector con el tamaño correcto.

    Procesa un frame con contenido aleatorio con MediaPipe Holistic y valida que el vector
    resultante contenga todos los keypoints esperados para cuerpo, cara y manos.

    Returns:
        None: Usa aserciones para validar el resultado.
    """
    with Holistic() as model:
        frame = _crear_frame_dummy()
        results = model.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        keypoints = extract_keypoints(results)
        assert isinstance(keypoints, np.ndarray)
        assert keypoints.shape[0] == (33 * 4 + 468 * 3 + 21 * 3 + 21 * 3)


def test_get_keypoints_devuelve_secuencia(tmp_path):
    """
    Verifica que `get_keypoints` procese todos los frames y devuelva una secuencia.

    Crea un directorio temporal con 3 imágenes y extrae los keypoints usando Holistic,
    comprobando que la longitud de la secuencia coincida con el número de imágenes.

    Args:
        tmp_path (Path): Ruta temporal de pytest.

    Returns:
        None: Usa aserciones para validar el resultado.
    """
    sample_dir = tmp_path / "sample_01"
    os.makedirs(sample_dir)
    for i in range(3):
        frame = _crear_frame_dummy()
        cv2.imwrite(str(sample_dir / f"frame_{i}.jpg"), frame)

    with Holistic() as model:
        sequence = get_keypoints(model, str(sample_dir))

    assert isinstance(sequence, np.ndarray)
    assert sequence.ndim == 2
    assert sequence.shape[0] == 3


def test_insert_keypoints_sequence():
    """
    Verifica que `insert_keypoints_sequence` agregue correctamente los datos al DataFrame.

    Se simula una secuencia de keypoints y se inserta en un DataFrame vacío, validando
    que las columnas `sample` y `frame` se llenen adecuadamente.

    Returns:
        None: Usa aserciones para validar el resultado.
    """
    df = pd.DataFrame(columns=["sample", "frame", "keypoints"])
    seq = np.random.rand(5, 1662)
    result = insert_keypoints_sequence(df, 1, seq)

    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == 5
    assert result["sample"].nunique() == 1
    assert all(result["frame"] == [1, 2, 3, 4, 5])


def test_group_keypoints_by_word_and_sample():
    """
    Verifica la agrupación correcta de secuencias por palabra y muestra.

    A partir de una lista simulada de registros de keypoints, se valida que
    `group_keypoints_by_word_and_sample` genere las secuencias correctas
    y que las etiquetas estén en el orden esperado.

    Returns:
        None: Usa aserciones para validar los resultados.
    """
    keypoints_data = [
        (b"abc", 1, 1, [0.1, 0.2]),
        (b"abc", 1, 2, [0.3, 0.4]),
        (b"def", 2, 1, [0.5, 0.6]),
        (b"def", 2, 2, [0.7, 0.8]),
    ]
    words_id = [b"abc", b"def"]

    sequences, labels = group_keypoints_by_word_and_sample(keypoints_data, words_id)

    assert len(sequences) == 2
    assert len(labels) == 2
    assert labels == [0, 1]
    assert all(isinstance(seq, list) for seq in sequences)
    assert all(isinstance(val, int) for val in labels)
