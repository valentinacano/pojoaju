"""
Pipeline completo para captura, normalización y extracción de keypoints desde cámara.

Este módulo permite:
1. Capturar muestras en tiempo real utilizando MediaPipe Holistic.
2. Normalizar la cantidad de frames por muestra.
3. Extraer los vectores de keypoints y guardarlos directamente en la base de datos PostgreSQL.

Es compatible tanto con ejecución en consola como desde una interfaz web Flask.
"""

import os, re, shutil
from ml.features.capture_samples import capture_samples_from_camera
from ml.features.capture_samples_video import capture_samples_from_video
from ml.features.normalize_samples import normalize_samples
from ml.features.create_keypoints import get_keypoints
from ml.utils.common_utils import create_folder
from app.database.database_utils import (
    insert_sample,
    insert_keypoints,
)


def create_samples_from_camera(word_name, root_path, debug_value=False):
    """
    Inicia la captura de muestras para una palabra desde la cámara.

    Crea la carpeta correspondiente y lanza el proceso de detección y captura usando MediaPipe Holistic.
    En modo consola (`debug=True`), ejecuta el flujo completo internamente.
    En modo Flask (`debug=False`), retorna un generador de imágenes codificadas JPEG para streaming.

    Args:
        word_name (str): Palabra que se desea grabar.
        root_path (str): Carpeta base donde se almacenarán las muestras por palabra.
        debug_value (bool): Indica si se ejecuta en consola (`True`) o en servidor Flask (`False`).

    Returns:
        Generator[bytes] | None: En modo Flask, retorna un generador de imágenes JPEG para streaming. En modo consola, no retorna nada.
    """
    word_path = os.path.join(root_path, word_name)
    create_folder(word_path)
    print(f"\n📸 Iniciando captura para la palabra: {word_name}")
    generator = capture_samples_from_camera(path=word_path, debug=debug_value)

    if debug_value:
        # Modo consola: consume el generador internamente
        for _ in generator:
            pass
    else:
        # Modo servidor (Flask): retorna el generador para streaming
        return generator


def save_keypoints(word_name, word_id, root_path):
    """
    Normaliza las muestras y extrae los keypoints para una palabra.

    Este pipeline ajusta la longitud de cada muestra a una cantidad fija de frames
    y luego guarda los vectores de keypoints extraídos en la base de datos.

    Args:
        word_name (str): Nombre de la palabra (debe coincidir con la carpeta de muestras).
        word_id (str | bytes): ID único de la palabra usado para la base de datos.
        root_path (str): Ruta donde se encuentran las carpetas de muestras.

    Returns:
        None: Esta función no retorna ningún valor. Inserta los datos procesados en la base de datos y elimina las carpetas temporales.
    """
    print("\n🚀 save_keypoints() fue llamado")

    # ✅ Validar formato del hash: 64 caracteres hexadecimales
    if isinstance(word_id, str):
        if not re.fullmatch(r"[0-9a-f]{64}", word_id):
            print(f"❌ word_id inválido: {word_id}")
            return
        try:
            word_id = bytes.fromhex(word_id)
            print("Se convirtio correctamente a hexa el word id")
        except ValueError:
            print("❌ Error al convertir word_id de hex a bytes.")
            return

    word_path = os.path.join(root_path, word_name)

    print(f"\n🌀 Normalizando muestras en: {word_path}")
    normalize_samples(word_path)

    sample_folders = sorted(
        [
            f
            for f in os.listdir(word_path)
            if os.path.isdir(os.path.join(word_path, f)) and f.startswith("sample_")
        ]
    )

    for folder in sample_folders:
        full_path = os.path.join(word_path, folder)
        from mediapipe.python.solutions.holistic import Holistic

        with Holistic() as model:
            keypoints_sequence = get_keypoints(model, full_path)

        if keypoints_sequence is None or len(keypoints_sequence) == 0:
            print(f"⚠️ No se generaron keypoints para {folder}, se omite.")
            continue

        sample_id = insert_sample(word_id)
        insert_keypoints(word_id, sample_id, keypoints_sequence)

        # Eliminar carpeta de muestra una vez procesada
        shutil.rmtree(full_path)

    print("\n✅ Proceso completado con éxito.")


def create_samples_from_video(word_name, root_path, video_path, debug_value=False):
    """
    Inicia la captura de muestras para una palabra desde la cámara.

    Crea la carpeta correspondiente y lanza el proceso de detección y captura usando MediaPipe Holistic.
    En modo consola (`debug=True`), ejecuta el flujo completo internamente.
    En modo Flask (`debug=False`), retorna un generador de imágenes codificadas JPEG para streaming.

    Args:
        word_name (str): Palabra que se desea grabar.
        root_path (str): Carpeta base donde se almacenarán las muestras por palabra.
        debug_value (bool): Indica si se ejecuta en consola (`True`) o en servidor Flask (`False`).

    Returns:
        Generator[bytes] | None: En modo Flask, retorna un generador de imágenes JPEG para streaming. En modo consola, no retorna nada.
    """

    word_path = os.path.join(root_path, word_name)
    create_folder(word_path)
    print(f"\n📸 Iniciando captura para la palabra: {word_name}")
    generator = capture_samples_from_video(
        path=word_path, video_path=video_path, debug=debug_value
    )

    if debug_value:
        # Modo consola: consume el generador internamente
        for _ in generator:
            pass
    else:
        # Modo servidor (Flask): retorna el generador para streaming
        return generator
