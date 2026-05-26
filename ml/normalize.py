"""
Normalización de secuencias — método único del proyecto.

REGLA FUNDAMENTAL:
    Esta función debe ser llamada tanto en training como en predicción.
    Si se usan métodos distintos, el modelo aprende con una distribución
    y predice con otra → baja accuracy garantizada.

Método: interpolación lineal para secuencias cortas, submuestreo
uniforme para secuencias largas.
"""

import os
import cv2
import shutil
import numpy as np

from app.config import MODEL_FRAMES


# ---------------------------------------------------------------------------
# Normalización de keypoints (usada en predicción en tiempo real)
# ---------------------------------------------------------------------------


def normalize_sequence(
    sequence: list | np.ndarray, target: int = MODEL_FRAMES
) -> np.ndarray:
    """
    Ajusta una secuencia de keypoints a exactamente `target` frames.

    - Si tiene menos frames: interpola linealmente entre frames existentes.
    - Si tiene más frames: submuestreo uniforme.
    - Si ya tiene exactamente `target`: retorna sin modificar.

    Este método es el ÚNICO que se debe usar en todo el proyecto,
    tanto en entrenamiento como en predicción.

    Args:
        sequence: lista o array de shape (n_frames, n_features).
        target: cantidad de frames deseada (default: MODEL_FRAMES).

    Returns:
        np.ndarray de shape (target, n_features).
    """
    seq = np.array(sequence)
    current = len(seq)

    if current == target:
        return seq

    indices = np.linspace(0, current - 1, target)
    result = []

    for i in indices:
        low = int(np.floor(i))
        high = int(np.ceil(i))
        weight = i - low

        if low == high:
            result.append(seq[low])
        else:
            interpolated = (1 - weight) * seq[low] + weight * seq[high]
            result.append(interpolated)

    return np.array(result)


# ---------------------------------------------------------------------------
# Normalización de frames de imagen (usada en captura → disco)
# ---------------------------------------------------------------------------


def normalize_frames(frames: list) -> list:
    """
    Ajusta una lista de frames (imágenes) a exactamente MODEL_FRAMES.

    Misma lógica que normalize_sequence pero para imágenes OpenCV.
    Se usa durante el pipeline de captura antes de extraer keypoints.

    Args:
        frames: lista de np.ndarray (imágenes BGR).

    Returns:
        lista de np.ndarray con exactamente MODEL_FRAMES imágenes.
    """
    current = len(frames)

    if current == MODEL_FRAMES:
        return frames

    indices = np.linspace(0, current - 1, MODEL_FRAMES)
    result = []

    for i in indices:
        low = int(np.floor(i))
        high = int(np.ceil(i))
        weight = i - low

        if low == high:
            result.append(frames[low])
        else:
            blended = cv2.addWeighted(frames[low], 1 - weight, frames[high], weight, 0)
            result.append(blended)

    return result


# ---------------------------------------------------------------------------
# Operaciones sobre disco (usadas en el pipeline de captura)
# ---------------------------------------------------------------------------


def read_frames_from_folder(folder: str) -> list:
    """
    Lee todos los frames .jpg de una carpeta, ordenados numéricamente.

    Args:
        folder: ruta a la carpeta.

    Returns:
        lista de np.ndarray (imágenes BGR).
    """
    files = sorted(
        [f for f in os.listdir(folder) if f.endswith(".jpg")],
        key=lambda x: int(os.path.splitext(x)[0].split("_")[-1]),
    )
    frames = []
    for fname in files:
        img = cv2.imread(os.path.join(folder, fname))
        if img is not None:
            frames.append(img)
    return frames


def save_frames_to_folder(folder: str, frames: list):
    """
    Guarda una lista de frames en una carpeta como frame_01.jpg, frame_02.jpg...

    Limpia el contenido previo antes de guardar.

    Args:
        folder: ruta destino.
        frames: lista de np.ndarray (imágenes BGR).
    """
    for item in os.listdir(folder):
        path = os.path.join(folder, item)
        if os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)

    for i, frame in enumerate(frames, start=1):
        cv2.imwrite(
            os.path.join(folder, f"frame_{i:02d}.jpg"),
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 85],
        )


def normalize_sample_folder(folder: str):
    """
    Lee, normaliza y sobreescribe los frames de una carpeta de muestra.

    Args:
        folder: ruta a la carpeta de una muestra individual.
    """
    frames = read_frames_from_folder(folder)
    if not frames:
        print(f"⚠️ Carpeta vacía, se omite: {folder}")
        return
    normalized = normalize_frames(frames)
    save_frames_to_folder(folder, normalized)


def normalize_word_folder(word_path: str):
    """
    Normaliza todas las muestras dentro de la carpeta de una palabra.

    Args:
        word_path: ruta que contiene subcarpetas sample_*/
    """
    sample_folders = sorted(
        [f for f in os.listdir(word_path) if os.path.isdir(os.path.join(word_path, f))]
    )

    print(f"🔄 Normalizando {len(sample_folders)} muestras en {word_path}...")

    for folder in sample_folders:
        normalize_sample_folder(os.path.join(word_path, folder))

    print("✅ Normalización completa.")
