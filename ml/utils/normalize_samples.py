import cv2
import numpy as np
import os
import shutil


def read_frames_from_directory(directory):
    """
    Lee y carga todos los frames .jpg desde un directorio.

    Ordena alfabéticamente los archivos .jpg dentro del directorio y los carga como imágenes en una lista.

    Args:
        directory (str): Ruta al directorio que contiene los frames en formato .jpg.

    Returns:
        list[numpy.ndarray]: Lista de imágenes cargadas desde el directorio.
    """
    frames = []
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".jpg"):
            frame = cv2.imread(os.path.join(directory, filename))
            frames.append(frame)
    return frames


def interpolate_frames(frames, target_frame_count=15):
    """
    Interpola una secuencia de imágenes para alcanzar una cantidad deseada de frames.

    Utiliza `cv2.addWeighted` para interpolar entre frames consecutivos en base a pesos calculados.

    Args:
        frames (list[numpy.ndarray]): Lista original de frames.
        target_frame_count (int): Cantidad de frames deseada.

    Returns:
        list[numpy.ndarray]: Lista de frames interpolados para que coincidan con el total deseado.
    """
    current_frame_count = len(frames)
    if current_frame_count == target_frame_count:
        return frames

    indices = np.linspace(0, current_frame_count - 1, target_frame_count)
    interpolated_frames = []
    for i in indices:
        lower_idx = int(np.floor(i))
        upper_idx = int(np.ceil(i))
        weight = i - lower_idx
        interpolated_frame = cv2.addWeighted(
            frames[lower_idx], 1 - weight, frames[upper_idx], weight, 0
        )
        interpolated_frames.append(interpolated_frame)

    return interpolated_frames


def clear_directory(directory):
    """
    Elimina todos los archivos y subdirectorios dentro de un directorio.

    Recorre recursivamente el contenido y lo elimina.

    Args:
        directory (str): Ruta al directorio que se desea limpiar.

    Returns:
        None: Esta función no retorna ningún valor.
    """
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)


def save_normalized_frames(directory, frames):
    """
    Guarda los frames normalizados como archivos JPEG comprimidos en un directorio.

    Cada imagen se guarda con nombre incremental (`frame_01.jpg`, `frame_02.jpg`, etc.) y calidad reducida.

    Args:
        directory (str): Ruta donde se guardarán los frames.
        frames (list[numpy.ndarray]): Lista de imágenes normalizadas a guardar.

    Returns:
        None: Esta función no retorna ningún valor.
    """
    for i, frame in enumerate(frames, start=1):
        cv2.imwrite(
            os.path.join(directory, f"frame_{i:02}.jpg"),
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 50],
        )


def normalize_frames(frames, target_frame_count=15):
    """
    Normaliza la cantidad de frames a una longitud fija.

    Si hay menos frames que los requeridos, interpola. Si hay más, hace un muestreo uniforme.
    Si ya coincide, retorna los mismos frames.

    Args:
        frames (list[numpy.ndarray]): Lista original de frames.
        target_frame_count (int): Cantidad deseada de frames.

    Returns:
        list[numpy.ndarray]: Lista de frames normalizada en longitud.
    """
    current_frame_count = len(frames)
    if current_frame_count < target_frame_count:
        return interpolate_frames(frames, target_frame_count)
    elif current_frame_count > target_frame_count:
        step = current_frame_count / target_frame_count
        indices = np.arange(0, current_frame_count, step).astype(int)[
            :target_frame_count
        ]
        return [frames[i] for i in indices]
    else:
        return frames
