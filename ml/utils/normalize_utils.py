"""
Utilidades para la normalización de secuencias de imágenes (frames).

Este módulo permite unificar la longitud de las muestras de video utilizadas para
entrenamiento, interpolando o recortando los frames para que todas tengan
la misma cantidad definida por `MODEL_FRAMES`.
"""


import os, cv2, shutil
import numpy as np

from app.config import MODEL_FRAMES


def read_frames_from_directory(directory):
    """
    Lee todos los frames `.jpg` desde un directorio dado.

    Ordena alfabéticamente los archivos `.jpg` y los carga como imágenes en memoria.

    Args:
        directory (str): Ruta que contiene los frames.

    Returns:
        list[numpy.ndarray]: Lista de imágenes leídas desde el directorio.
    """
    frames = []
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".jpg"):
            frame = cv2.imread(os.path.join(directory, filename))
            if frame is not None:
                frames.append(frame)
    return frames


def interpolate_frames(frames):
    """
    Interpola una lista de frames para alcanzar `MODEL_FRAMES`.

    Genera nuevos frames mediante interpolación lineal entre los frames existentes
    usando `cv2.addWeighted`. Es útil cuando hay menos frames que los necesarios.

    Args:
        frames (list[numpy.ndarray]): Lista de imágenes original.

    Returns:
        list[numpy.ndarray]: Lista con `MODEL_FRAMES` frames interpolados.
    """
    current = len(frames)
    if current == MODEL_FRAMES:
        return frames

    indices = np.linspace(0, current - 1, MODEL_FRAMES)
    interpolated = []
    for i in indices:
        low = int(np.floor(i))
        high = int(np.ceil(i))
        weight = i - low
        interpolated.append(
            cv2.addWeighted(frames[low], 1 - weight, frames[high], weight, 0)
        )
    return interpolated


def normalize_frames(frames):
    """
    Normaliza una secuencia de frames a una cantidad fija.

    Si hay menos frames que los deseados, interpola. Si hay más, recorta uniformemente.
    Si la cantidad coincide, se devuelve sin modificar.

    Args:
        frames (list[numpy.ndarray]): Lista original de imágenes.

    Returns:
        list[numpy.ndarray]: Lista de frames normalizada en longitud.
    """
    current = len(frames)
    if current < MODEL_FRAMES:
        return interpolate_frames(frames)
    elif current > MODEL_FRAMES:
        step = current / MODEL_FRAMES
        indices = np.arange(0, current, step).astype(int)[:MODEL_FRAMES]
        return [frames[i] for i in indices]
    else:
        return frames


def clear_directory(directory):
    """
    Elimina todos los archivos y subdirectorios dentro de un directorio.

    Esta función borra recursivamente el contenido del directorio, dejando su interior vacío.

    Args:
        directory (str): Ruta al directorio que se desea limpiar.

    Returns:
        None: No retorna ningún valor.
    """
    for item in os.listdir(directory):
        path = os.path.join(directory, item)
        if os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)


def save_normalized_frames(directory, frames):
    """
    Guarda una lista de frames como archivos JPEG comprimidos en un directorio.

    Cada frame se guarda con nombre incremental (`frame_01.jpg`, `frame_02.jpg`, etc.)
    con calidad de compresión reducida para ahorrar espacio.

    Args:
        directory (str): Ruta donde se guardarán los archivos.
        frames (list[numpy.ndarray]): Lista de imágenes a guardar.

    Returns:
        None: Esta función no retorna ningún valor.
    """
    for i, frame in enumerate(frames, start=1):
        filename = f"frame_{i:02}.jpg"
        cv2.imwrite(
            os.path.join(directory, filename),
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 50],
        )
