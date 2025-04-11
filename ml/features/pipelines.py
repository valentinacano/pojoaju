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


def create_samples_from_camera(
    word_name, root_path, keypoints_path, target_frame_count=15
):
    """
    Captura, normaliza y extrae keypoints desde cámara para una palabra dada.

    Args:
        word_name (str): Palabra que se va a grabar.
        root_path (str): Carpeta base donde se guardarán los frames por palabra.
        keypoints_path (str): Ruta del archivo `.h5` de salida.
        target_frame_count (int): Número de frames a normalizar por muestra.

    Returns:
        None
    """
    word_path = os.path.join(root_path, word_name)
    hdf_path = os.path.join(keypoints_path, f"{word_name}.h5")

    print(f"\n📸 Iniciando captura para la palabra: {word_name}")
    capture_samples_from_camera(word_path)

    print(f"\n🌀 Normalizando muestras en: {word_path}")
    normalize_samples(word_path, target_frame_count)

    print(f"\n🎯 Extrayendo keypoints y guardando en: {hdf_path}")
    create_folder(keypoints_path)
    create_keypoints(word_name, root_path, hdf_path)

    print("\n✅ Proceso completado con éxito.")
