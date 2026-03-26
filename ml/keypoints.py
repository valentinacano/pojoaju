"""
Extracción de keypoints con MediaPipe Holistic.

Este módulo es usado tanto en captura como en predicción.
Garantiza que el vector de features sea siempre idéntico (1662 valores).
"""

import os
import cv2
import numpy as np
from mediapipe.python.solutions.holistic import Holistic


def run_mediapipe(frame: np.ndarray, model) -> object:
    """
    Procesa un frame con MediaPipe.

    Args:
        frame: imagen BGR (numpy array).
        model: instancia activa de Holistic.

    Returns:
        Objeto results de MediaPipe.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = model.process(rgb)
    rgb.flags.writeable = True
    return results


def extract_keypoints(results) -> np.ndarray:
    """
    Extrae todos los landmarks detectados y los concatena en un vector.

    Estructura del vector (1662 valores):
        - Pose:       33 landmarks * 4 valores (x, y, z, visibility) = 132
        - Face:      468 landmarks * 3 valores (x, y, z)             = 1404
        - Left hand:  21 landmarks * 3 valores (x, y, z)             = 63
        - Right hand: 21 landmarks * 3 valores (x, y, z)             = 63

    Si alguna parte no es detectada, se rellena con ceros.

    Args:
        results: objeto devuelto por MediaPipe Holistic.

    Returns:
        np.ndarray de shape (1662,).
    """
    pose = (
        np.array([[lm.x, lm.y, lm.z, lm.visibility]
                  for lm in results.pose_landmarks.landmark]).flatten()
        if results.pose_landmarks else np.zeros(33 * 4)
    )
    face = (
        np.array([[lm.x, lm.y, lm.z]
                  for lm in results.face_landmarks.landmark]).flatten()
        if results.face_landmarks else np.zeros(468 * 3)
    )
    lh = (
        np.array([[lm.x, lm.y, lm.z]
                  for lm in results.left_hand_landmarks.landmark]).flatten()
        if results.left_hand_landmarks else np.zeros(21 * 3)
    )
    rh = (
        np.array([[lm.x, lm.y, lm.z]
                  for lm in results.right_hand_landmarks.landmark]).flatten()
        if results.right_hand_landmarks else np.zeros(21 * 3)
    )
    return np.concatenate([pose, face, lh, rh])


def has_hand(results) -> bool:
    """
    Retorna True si al menos una mano fue detectada.

    Args:
        results: objeto devuelto por MediaPipe Holistic.

    Returns:
        bool
    """
    return (
        results.left_hand_landmarks is not None
        or results.right_hand_landmarks is not None
    )


def extract_keypoints_from_folder(folder_path: str, model) -> np.ndarray:
    """
    Extrae keypoints de todos los frames en una carpeta.

    Lee los archivos .jpg ordenados numéricamente, procesa cada uno
    con MediaPipe y retorna la secuencia completa.

    Args:
        folder_path: ruta a la carpeta con frames .jpg.
        model: instancia activa de Holistic.

    Returns:
        np.ndarray de shape (n_frames, 1662).
    """
    files = sorted(
        [f for f in os.listdir(folder_path) if f.endswith(".jpg")],
        key=lambda x: int(os.path.splitext(x)[0].split("_")[-1])
    )
    sequence = []
    for fname in files:
        frame = cv2.imread(os.path.join(folder_path, fname))
        if frame is not None:
            results = run_mediapipe(frame, model)
            sequence.append(extract_keypoints(results))

    return np.array(sequence)