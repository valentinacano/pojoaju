"""
Funciones para la extracción y manejo de keypoints desde MediaPipe.

Permite convertir secuencias de imágenes en vectores numéricos de keypoints,
y guardar la información asociada a muestras en un DataFrame estructurado.
"""

import os
import cv2
import numpy as np
import pandas as pd
from ml.utils.common_utils import mediapipe_detection


def extract_keypoints(results):
    """
    Extrae los keypoints detectados por MediaPipe desde un resultado.

    Concatena los landmarks de cuerpo, rostro, mano izquierda y mano derecha
    en un único vector. Si alguna parte no fue detectada, se rellena con ceros.

    Args:
        results: Objeto devuelto por el modelo de MediaPipe tras procesar una imagen.

    Returns:
        numpy.ndarray: Vector unificado que representa todos los keypoints.
    """
    pose = (
        np.array(
            [[r.x, r.y, r.z, r.visibility] for r in results.pose_landmarks.landmark]
        ).flatten()
        if results.pose_landmarks
        else np.zeros(33 * 4)
    )
    face = (
        np.array([[r.x, r.y, r.z] for r in results.face_landmarks.landmark]).flatten()
        if results.face_landmarks
        else np.zeros(468 * 3)
    )
    lh = (
        np.array(
            [[r.x, r.y, r.z] for r in results.left_hand_landmarks.landmark]
        ).flatten()
        if results.left_hand_landmarks
        else np.zeros(21 * 3)
    )
    rh = (
        np.array(
            [[r.x, r.y, r.z] for r in results.right_hand_landmarks.landmark]
        ).flatten()
        if results.right_hand_landmarks
        else np.zeros(21 * 3)
    )
    return np.concatenate([pose, face, lh, rh])


def get_keypoints(model, sample_path):
    """
    Extrae keypoints de todos los frames en una carpeta de muestra.

    Procesa cada imagen usando el modelo de MediaPipe y extrae los keypoints
    correspondientes. Devuelve una secuencia completa.

    Args:
        model: Modelo de MediaPipe ya inicializado.
        sample_path (str): Ruta de la carpeta que contiene los frames.

    Returns:
        numpy.ndarray: Secuencia de vectores de keypoints.
    """
    keypoints_sequence = []
    for img_name in sorted(os.listdir(sample_path)):
        img_path = os.path.join(sample_path, img_name)
        frame = cv2.imread(img_path)
        if frame is not None:
            results = mediapipe_detection(frame, model)
            keypoints_sequence.append(extract_keypoints(results))
    return np.array(keypoints_sequence)


def insert_keypoints_sequence(df, sample_id, keypoints_sequence):
    """
    Inserta la secuencia de keypoints en un DataFrame con su ID de muestra y frame.

    Cada frame se inserta como una fila con el número de muestra y el índice de frame correspondiente.

    Args:
        df (pandas.DataFrame): DataFrame donde se insertarán los nuevos registros.
        sample_id (int): Identificador de la muestra actual.
        keypoints_sequence (numpy.ndarray): Secuencia de vectores de keypoints.

    Returns:
        pandas.DataFrame: El DataFrame original actualizado con la nueva muestra.
    """
    records = [
        {"sample": sample_id, "frame": i + 1, "keypoints": [kp]}
        for i, kp in enumerate(keypoints_sequence)
    ]
    return pd.concat([df, pd.DataFrame(records)], ignore_index=True)
