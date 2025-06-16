"""
Tests para la función de normalización de muestras.

Este módulo verifica que las muestras de video se normalicen correctamente
a una longitud fija (`MODEL_FRAMES`) y que los archivos resultantes conserven
el formato y las dimensiones esperadas.
"""

import os
import numpy as np
import pytest
import cv2
from ml.features.normalize_samples import normalize_samples
from app.config import MODEL_FRAMES


@pytest.fixture
def dummy_word_folder(tmp_path):
    """
    Crea una estructura de carpetas con muestras de frames en formato .jpg para testeo.

    Esta fixture genera dos carpetas de muestras con diferente cantidad de frames,
    simulando una situación real para testear la normalización.

    Args:
        tmp_path (Path): Carpeta temporal provista por pytest.

    Returns:
        Path: Ruta a la carpeta raíz de la palabra con las muestras creadas.
    """
    word_folder = tmp_path / "hola"
    os.makedirs(word_folder)

    sample1 = word_folder / "sample_01"
    sample2 = word_folder / "sample_02"
    sample1.mkdir()
    sample2.mkdir()

    # sample1 tiene 10 frames
    for i in range(10):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(sample1 / f"{i+1}.jpg"), frame)

    # sample2 tiene 5 frames
    for i in range(5):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(sample2 / f"{i+1}.jpg"), frame)

    return word_folder


def test_normalize_samples_lengths(dummy_word_folder):
    """
    Testea que todas las muestras queden con la misma longitud tras normalizar.

    Ejecuta la función `normalize_samples` sobre las carpetas creadas y verifica
    que la cantidad de archivos por muestra coincida con el valor de `MODEL_FRAMES`.

    Args:
        dummy_word_folder (Path): Fixture con muestras pre-generadas.

    Returns:
        None: Utiliza aserciones para validar resultados.
    """
    normalize_samples(str(dummy_word_folder))

    sample1 = dummy_word_folder / "sample_01"
    sample2 = dummy_word_folder / "sample_02"

    files1 = sorted(sample1.glob("*.jpg"))
    files2 = sorted(sample2.glob("*.jpg"))

    assert (
        len(files1) == len(files2) == MODEL_FRAMES
    ), f"Se esperaban {MODEL_FRAMES} frames, pero se obtuvieron {len(files1)} y {len(files2)}"


def test_normalize_samples_overwrite(dummy_word_folder):
    """
    Verifica que la función sobreescribe correctamente los archivos originales.

    Ejecuta la normalización y revisa que los archivos resultantes tengan
    el formato y dimensiones esperadas.

    Args:
        dummy_word_folder (Path): Fixture con muestras pre-generadas.

    Returns:
        None: Utiliza aserciones para validar resultados.
    """
    normalize_samples(str(dummy_word_folder))
    sample1 = dummy_word_folder / "sample_01"
    data_path = next(sample1.glob("*.jpg"))
    data = cv2.imread(str(data_path))
    assert data.shape == (
        480,
        640,
        3,
    ), "Los archivos normalizados no tienen el shape esperado"
