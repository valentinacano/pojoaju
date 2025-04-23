"""
Funciones para visualizar y guardar imágenes con keypoints.

Dibuja landmarks detectados por MediaPipe y guarda secuencias de frames.
"""

import os
import cv2
from mediapipe.python.solutions.holistic import (
    FACEMESH_CONTOURS,
    POSE_CONNECTIONS,
    HAND_CONNECTIONS,
)
from mediapipe.python.solutions.drawing_utils import draw_landmarks, DrawingSpec


def draw_keypoints(image, results):
    """
    Dibuja los keypoints detectados por MediaPipe sobre una imagen.

    Esta función renderiza los landmarks de rostro, cuerpo y ambas manos, utilizando
    colores personalizados para cada tipo de conexión.

    Args:
        image (np.ndarray): Imagen original donde se dibujarán los keypoints.
        results: Resultados devueltos por MediaPipe tras procesar la imagen.

    Returns:
        None: Esta función no retorna ningún valor, modifica la imagen directamente.
    """
    draw_landmarks(
        image,
        results.face_landmarks,
        FACEMESH_CONTOURS,
        DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
        DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1),
    )
    draw_landmarks(
        image,
        results.pose_landmarks,
        POSE_CONNECTIONS,
        DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
        DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2),
    )
    draw_landmarks(
        image,
        results.left_hand_landmarks,
        HAND_CONNECTIONS,
        DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
        DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2),
    )
    draw_landmarks(
        image,
        results.right_hand_landmarks,
        HAND_CONNECTIONS,
        DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
        DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
    )


def save_frames(frames, output_folder):
    """
    Guarda una secuencia de imágenes como archivos numerados en una carpeta.

    Convierte los frames de BGR a BGRA y los guarda como archivos JPEG secuenciales
    en el directorio especificado.

    Args:
        frames (list[np.ndarray]): Lista de imágenes en formato BGR.
        output_folder (str): Ruta al directorio donde se guardarán los archivos.

    Returns:
        None: Esta función no retorna ningún valor. Guarda archivos en disco.
    """
    for i, frame in enumerate(frames, start=1):
        path = os.path.join(output_folder, f"{i}.jpg")
        cv2.imwrite(path, cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA))
