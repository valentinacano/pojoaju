"""
Tests para captura de muestras simuladas (`_save_sample`).

Este módulo contiene pruebas unitarias para validar el correcto funcionamiento
de la función `_save_sample`, encargada de recortar y guardar secuencias de frames
como imágenes numeradas dentro de una carpeta con timestamp.

Las pruebas se realizan sobre datos simulados y usan carpetas temporales.
"""

import os
import tempfile
import numpy as np

from ml.features.capture_samples import _save_sample


def _crear_frames_dummy(n=10, shape=(480, 640, 3)):
    """
    Crea una lista de frames negros para simular capturas.

    Esta utilidad genera arrays de imágenes en negro para ser utilizados
    en pruebas sin depender de una cámara real.

    Args:
        n (int): Cantidad de frames a generar.
        shape (tuple): Dimensiones de cada frame (alto, ancho, canales).

    Returns:
        list[np.ndarray]: Lista de frames negros simulados.
    """
    return [np.zeros(shape, dtype=np.uint8) for _ in range(n)]


def test_save_sample_crea_carpeta_con_imagenes():
    """
    Verifica que `_save_sample` cree correctamente una carpeta de muestra
    y guarde en ella los frames recortados en formato .jpg.

    Este test utiliza una carpeta temporal para guardar los resultados y
    valida:
    - Que se crea una única subcarpeta.
    - Que se guardan 7 imágenes (10 - 1 margin - 2 delay).
    - Que los archivos tienen nombres secuenciales correctos.

    Returns:
        None: Utiliza aserciones para validar el comportamiento.
    """
    frames = _crear_frames_dummy(10)
    with tempfile.TemporaryDirectory() as tmpdir:
        _save_sample(frames, tmpdir, margin_frames=1, delay_frames=2)

        subfolders = [
            f for f in os.listdir(tmpdir) if os.path.isdir(os.path.join(tmpdir, f))
        ]
        assert len(subfolders) == 1, "Debe crearse una sola carpeta de muestra."

        sample_folder = os.path.join(tmpdir, subfolders[0])
        saved_images = sorted(
            f for f in os.listdir(sample_folder) if f.endswith(".jpg")
        )

        assert len(saved_images) == 7, "Se esperaban 7 imágenes (10 - 1 - 2)."
        assert saved_images[0] == "1.jpg" and saved_images[-1] == "7.jpg"
        for img in saved_images:
            path = os.path.join(sample_folder, img)
            assert os.path.exists(path), f"La imagen {img} no fue encontrada."
