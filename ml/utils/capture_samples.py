"""
Funciones para visualizar y guardar imágenes con keypoints detectados.

Este módulo permite dibujar landmarks detectados por MediaPipe en una imagen
y guardar una secuencia de frames en una carpeta.
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
    Dibuja los keypoints detectados por MediaPipe en una imagen.

    Esta función renderiza los landmarks de rostro, cuerpo y ambas manos,
    utilizando distintos colores para cada tipo de landmark.

    Args:
        image (numpy.ndarray): Imagen sobre la cual se dibujarán los keypoints.
        results: Resultado devuelto por el modelo de MediaPipe.

    Returns:
        None: Esta función modifica la imagen en lugar de devolver un valor.
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
    Guarda una secuencia de imágenes como archivos .jpg numerados.

    Recorre todos los frames y los guarda en la carpeta destino con nombres secuenciales.

    Args:
        frames (list): Lista de imágenes en formato numpy (BGR).
        output_folder (str): Ruta a la carpeta donde se guardarán los archivos.

    Returns:
        None: Esta función no retorna ningún valor.
    """
    for num_frame, frame in enumerate(frames):
        frame_path = os.path.join(output_folder, f"{num_frame + 1}.jpg")
        cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA))
