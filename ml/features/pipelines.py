"""
Pipeline de creación de muestras desde cámara.

Este flujo realiza la captura de muestras con MediaPipe Holistic, las normaliza
y genera los vectores de keypoints para su uso posterior en modelos de IA.
"""

import os
from ml.features.capture_samples import capture_samples_from_camera
from ml.features.normalize_samples import normalize_samples
from ml.features.create_keypoints import create_keypoints
from ml.utils.common_utils import create_folder


def create_samples_from_camera(word_name, root_path, target_frame_count=15):
    """
    Inicia la captura de muestras para una palabra desde la cámara.

    Crea la carpeta correspondiente, lanza la detección con MediaPipe y devuelve
    un generador de frames en formato JPEG para transmisión en vivo (streaming).

    Args:
        word_name (str): Palabra que se desea grabar.
        root_path (str): Carpeta base donde se almacenarán los samples.
        target_frame_count (int): Número de frames deseado por muestra (no usado directamente aquí, pero útil para coherencia del pipeline).

    Returns:
        Generator[bytes]: Flujo de imágenes codificadas en JPEG para visualización en tiempo real.
    """
    word_path = os.path.join(root_path, word_name)
    create_folder(word_path)
    print(f"\n📸 Iniciando captura para la palabra: {word_name}")
    result = capture_samples_from_camera(word_path, debug=False)
    return result


def save_samples(
    word_name, root_path, keypoints_path, target_frame_count=15
):
    """
    Normaliza y extrae keypoints de las muestras grabadas para una palabra.

    Este proceso ajusta la cantidad de frames por muestra a un valor fijo, y luego
    genera un archivo HDF5 con los vectores de keypoints listos para entrenamiento.

    Args:
        word_name (str): Palabra correspondiente a las muestras capturadas.
        root_path (str): Carpeta base donde están los frames capturados.
        keypoints_path (str): Ruta donde se guardará el archivo `.h5` con los keypoints.
        target_frame_count (int): Número de frames a los que se normalizarán las muestras.

    Returns:
        None: Esta función no retorna ningún valor, pero genera un archivo `.h5` con los resultados.
    """
    word_path = os.path.join(root_path, word_name)
    keypoints_path = os.path.join(keypoints_path, f"{word_name}.h5")

    print(f"\n🌀 Normalizando muestras en: {word_path}")
    normalize_samples(word_path, target_frame_count)

    print(f"\n🎯 Extrayendo keypoints y guardando en: {keypoints_path}")
    create_folder(keypoints_path)
    create_keypoints(word_name, root_path, keypoints_path)

    print("\n✅ Proceso completado con éxito.")
