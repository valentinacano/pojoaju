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
    Lee todos los frames .jpg desde un directorio dado.

    Args:
        directory (str): Ruta que contiene los frames.

    Returns:
        list[numpy.ndarray]: Lista de imágenes leídas.
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
    Interpola frames para alcanzar una cantidad deseada.

    Usa `cv2.addWeighted` para generar nuevos frames interpolados.

    Args:
        frames (list[numpy.ndarray]): Lista original.
        target_frame_count (int): Longitud deseada.

    Returns:
        list[numpy.ndarray]: Frames interpolados.
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
    Normaliza una secuencia de frames a una longitud fija.

    Interpola si hay menos, recorta si hay más.

    Args:
        frames (list[numpy.ndarray]): Lista original.
        target_frame_count (int): Longitud deseada.

    Returns:
        list[numpy.ndarray]: Lista normalizada.
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
    Elimina todos los archivos de un directorio.

    Args:
        directory (str): Ruta al directorio.

    Returns:
        None
    """
    for item in os.listdir(directory):
        path = os.path.join(directory, item)
        if os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)


def save_normalized_frames(directory, frames):
    """
    Guarda los frames como archivos JPEG comprimidos.

    Args:
        directory (str): Ruta destino.
        frames (list[numpy.ndarray]): Lista de imágenes.

    Returns:
        None
    """
    for i, frame in enumerate(frames, start=1):
        filename = f"frame_{i:02}.jpg"
        cv2.imwrite(
            os.path.join(directory, filename),
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 50],
        )
