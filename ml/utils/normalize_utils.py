"""
Funciones de utilidades para normalización de frames.

Incluye lectura, interpolación, recorte y guardado de frames normalizados
para que todas las muestras tengan la misma longitud.
"""

import os
import cv2
import numpy as np
import shutil


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


def interpolate_frames(frames, target_frame_count=15):
    """
    Interpola una lista de frames para alcanzar una longitud fija.

    Utiliza interpolación lineal con `cv2.addWeighted` entre pares de imágenes.

    Args:
        frames (list[numpy.ndarray]): Lista de imágenes original.
        target_frame_count (int): Cantidad deseada de frames.

    Returns:
        list[numpy.ndarray]: Lista con los frames interpolados.
    """
    current = len(frames)
    if current == target_frame_count:
        return frames

    indices = np.linspace(0, current - 1, target_frame_count)
    interpolated = []
    for i in indices:
        low = int(np.floor(i))
        high = int(np.ceil(i))
        weight = i - low
        interpolated.append(
            cv2.addWeighted(frames[low], 1 - weight, frames[high], weight, 0)
        )
    return interpolated


def normalize_frames(frames, target_frame_count=15):
    """
    Normaliza una secuencia de frames a una cantidad fija.

    Si hay menos frames que los deseados, interpola. Si hay más, recorta uniformemente.
    Si la cantidad coincide, se devuelve sin modificar.

    Args:
        frames (list[numpy.ndarray]): Lista original de imágenes.
        target_frame_count (int): Longitud deseada.

    Returns:
        list[numpy.ndarray]: Lista de frames normalizada en longitud.
    """
    current = len(frames)
    if current < target_frame_count:
        return interpolate_frames(frames, target_frame_count)
    elif current > target_frame_count:
        step = current / target_frame_count
        indices = np.arange(0, current, step).astype(int)[:target_frame_count]
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
