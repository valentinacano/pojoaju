"""
Orquestador del pipeline de ML.

Coordina captura → normalización → extracción de keypoints → guardado en BD.
Es el único módulo que Flask necesita importar para el flujo de entrenamiento.
"""

import os
import shutil
from mediapipe.python.solutions.holistic import Holistic

from app.config import FRAMES_PATH, EXPORTS_PATH
from app.database.queries import insert_sample, insert_keypoints, word_to_id
from ml.capture import capture_from_camera, capture_from_video, stop_capture
from ml.normalize import normalize_word_folder
from ml.keypoints import extract_keypoints_from_folder
from ml.train import train
from ml.predict import predict_stream
from ml.evaluate import generate_confusion_matrix, get_top_confusions


# ---------------------------------------------------------------------------
# Captura
# ---------------------------------------------------------------------------


def start_capture_camera(word: str, debug: bool = False, camera_index: int = 0):
    """
    Inicia la captura de muestras para una palabra desde la cámara.

    Args:
        word: nombre de la palabra.
        debug: True = consola, False = generador JPEG para Flask.
        camera_index: índice de la cámara.

    Returns:
        Generador JPEG si debug=False, None si debug=True.
    """
    path = os.path.join(FRAMES_PATH, word.strip().lower())
    return capture_from_camera(path, debug=debug, camera_index=camera_index)


def start_capture_video(word: str, video_path: str):
    """
    Procesa un video y extrae muestras para una palabra.

    Args:
        word: nombre de la palabra.
        video_path: ruta al archivo de video.
    """
    path = os.path.join(FRAMES_PATH, word.strip().lower())
    capture_from_video(video_path, path)


def stop_capture_camera():
    """Detiene la captura de cámara en curso."""
    stop_capture()


# ---------------------------------------------------------------------------
# Procesamiento de muestras → BD
# ---------------------------------------------------------------------------


def process_and_save(word: str, word_id_hex: str):
    """
    Normaliza las muestras capturadas, extrae keypoints y los guarda en la BD.

    Pasos:
        1. Normaliza los frames de cada muestra a MODEL_FRAMES
        2. Extrae los 1662 keypoints por frame con MediaPipe
        3. Inserta sample + keypoints en PostgreSQL
        4. Elimina las carpetas temporales de frames

    Args:
        word: nombre de la palabra.
        word_id_hex: word_id en formato hex string (64 chars).
    """
    print(f"\n🚀 Procesando muestras para: {word}")

    # Convertir hex → bytes
    try:
        wid = bytes.fromhex(word_id_hex)
    except ValueError:
        print(f"❌ word_id_hex inválido: {word_id_hex}")
        return

    word_path = os.path.join(FRAMES_PATH, word.strip().lower())
    if not os.path.exists(word_path):
        print(f"❌ No existe la carpeta: {word_path}")
        return

    # 1. Normalizar frames
    normalize_word_folder(word_path)

    # 2. Extraer keypoints e insertar en BD
    sample_folders = sorted(
        [f for f in os.listdir(word_path) if os.path.isdir(os.path.join(word_path, f))]
    )

    if not sample_folders:
        print("⚠️ No se encontraron carpetas de muestra.")
        return

    with Holistic() as model:
        for folder in sample_folders:
            folder_path = os.path.join(word_path, folder)
            sequence = extract_keypoints_from_folder(folder_path, model)

            if len(sequence) == 0:
                print(f"⚠️ Sin keypoints en {folder}, se omite.")
                continue

            sample_id = insert_sample(wid)
            insert_keypoints(wid, sample_id, sequence)

            # Eliminar carpeta procesada
            shutil.rmtree(folder_path)

    print(f"\n✅ Procesamiento completo para '{word}'.")


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------


def run_training(epochs: int = 300) -> dict:
    """Ejecuta el pipeline de entrenamiento y retorna métricas."""
    return train(epochs=epochs)


# ---------------------------------------------------------------------------
# Predicción
# ---------------------------------------------------------------------------


def run_predict_stream(camera_index: int = 0):
    """Retorna el generador de predicción para Flask."""
    return predict_stream(camera_index=camera_index)


# ---------------------------------------------------------------------------
# Evaluación
# ---------------------------------------------------------------------------


def run_evaluation() -> dict:
    """
    Genera la matriz de confusión y retorna métricas y confusiones principales.
    """
    cm, y_val, y_pred, metrics = generate_confusion_matrix()
    top = get_top_confusions(cm, metrics["labels"])

    return {
        "image_path": "/static/confusion/confusion_matrix.png",
        "accuracy": f"{metrics['accuracy']:.2%}",
        "n_classes": metrics["n_classes"],
        "n_samples": metrics["n_samples"],
        "labels": metrics["labels"],
        "top_confusions": [{"real": r, "predicted": p, "count": c} for r, p, c in top],
        "report": metrics["report"],
    }
