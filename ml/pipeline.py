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


def start_capture_video(word: str, video_path: str, sample_count: int = 1):
    path = os.path.join(FRAMES_PATH, word.strip().lower())
    print(f"🚀 Procesando {sample_count} muestras para: {word}")
    capture_from_video(video_path, path, sample_count)


def stop_capture_camera():
    """Detiene la captura de cámara en curso."""
    stop_capture()


# ---------------------------------------------------------------------------
# Procesamiento de muestras → BD
# ---------------------------------------------------------------------------

import threading

_progress_store: dict[str, list[str]] = {}
_progress_done: dict[str, bool] = {}


def process_and_save(word: str, word_id_hex: str):
    key = f"{word_id_hex}_{word}"
    _progress_store[key] = []
    _progress_done[key] = False

    def emit(msg: str):
        _progress_store[key].append(msg)
        print(msg)

    def run():
        emit(f"🚀 Iniciando procesamiento para '{word}'...")

        try:
            wid = bytes.fromhex(word_id_hex)
        except ValueError:
            emit(f"❌ word_id_hex inválido: {word_id_hex}")
            _progress_done[key] = True
            return

        word_path = os.path.join(FRAMES_PATH, word.strip().lower())
        if not os.path.exists(word_path):
            emit(f"❌ No existe la carpeta: {word_path}")
            _progress_done[key] = True
            return

        emit("🔄 Normalizando frames...")
        normalize_word_folder(word_path)
        emit("✅ Normalización completa.")

        sample_folders = sorted(
            [
                f
                for f in os.listdir(word_path)
                if os.path.isdir(os.path.join(word_path, f))
            ]
        )

        if not sample_folders:
            emit("⚠️ No se encontraron carpetas de muestra.")
            _progress_done[key] = True
            return

        total = len(sample_folders)
        emit(f"📦 Procesando {total} muestras...")

        with Holistic() as model:
            for i, folder in enumerate(sample_folders, 1):
                folder_path = os.path.join(word_path, folder)
                sequence = extract_keypoints_from_folder(folder_path, model)

                if len(sequence) == 0:
                    emit(f"⚠️ Sin keypoints en {folder}, se omite.")
                    continue

                sample_id = insert_sample(wid)
                insert_keypoints(wid, sample_id, sequence)
                shutil.rmtree(folder_path)
                emit(f"PROGRESS:{i}/{total}")

        emit(f"✅ Procesamiento completo para '{word}'.")
        _progress_done[key] = True

    threading.Thread(target=run, daemon=True).start()


def get_progress(word: str, word_id_hex: str):
    key = f"{word_id_hex}_{word}"
    return _progress_store.get(key, []), _progress_done.get(key, False)


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
