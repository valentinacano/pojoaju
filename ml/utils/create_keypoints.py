"""
Funciones para la extracción y manejo de keypoints desde resultados de MediaPipe.

Este módulo permite obtener representaciones vectoriales del cuerpo, rostro y manos
a partir de una secuencia de imágenes o frames procesados por MediaPipe.
"""

import os
import cv2
import numpy as np
import pandas as pd
from ml.utils.general import mediapipe_detection


def extract_keypoints(results):
    """
    Extrae los keypoints detectados por MediaPipe y los concatena en un solo vector.

    Extrae los landmarks del cuerpo, rostro, mano izquierda y mano derecha (si existen).
    Si alguno no es detectado, se completa con ceros para mantener la estructura fija.

    Args:
        results: Objeto de resultados devuelto por MediaPipe tras procesar una imagen.

    Returns:
        numpy.ndarray: Vector unificado con todos los keypoints extraídos.
    """
    pose = (
        np.array(
            [
                [res.x, res.y, res.z, res.visibility]
                for res in results.pose_landmarks.landmark
            ]
        ).flatten()
        if results.pose_landmarks
        else np.zeros(33 * 4)
    )
    face = (
        np.array(
            [[res.x, res.y, res.z] for res in results.face_landmarks.landmark]
        ).flatten()
        if results.face_landmarks
        else np.zeros(468 * 3)
    )
    lh = (
        np.array(
            [[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]
        ).flatten()
        if results.left_hand_landmarks
        else np.zeros(21 * 3)
    )
    rh = (
        np.array(
            [[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]
        ).flatten()
        if results.right_hand_landmarks
        else np.zeros(21 * 3)
    )
    return np.concatenate([pose, face, lh, rh])


def get_keypoints(model, sample_path):
    """
    Obtiene la secuencia de keypoints de una muestra de imágenes.

    Recorre todas las imágenes en una carpeta, ejecuta la detección con MediaPipe
    y extrae los keypoints de cada frame, construyendo una secuencia.

    Args:
        model: Modelo de MediaPipe cargado.
        sample_path (str): Ruta a la carpeta que contiene las imágenes.

    Returns:
        numpy.ndarray: Secuencia de vectores de keypoints por frame.
    """
    kp_seq = np.array([])
    for img_name in os.listdir(sample_path):
        img_path = os.path.join(sample_path, img_name)
        frame = cv2.imread(img_path)
        results = mediapipe_detection(frame, model)
        kp_frame = extract_keypoints(results)
        kp_seq = np.concatenate(
            [kp_seq, [kp_frame]] if kp_seq.size > 0 else [[kp_frame]]
        )
    return kp_seq


def insert_keypoints_sequence(df, n_sample, kp_seq):
    """
    Inserta una secuencia de keypoints al DataFrame con información de muestra y frame.

    Por cada frame, agrega una fila nueva con los keypoints, el número de muestra y
    el índice del frame dentro del DataFrame original.

    Args:
        df (pandas.DataFrame): DataFrame donde se insertarán los datos.
        n_sample (int): Número identificador de la muestra.
        kp_seq (numpy.ndarray): Secuencia de keypoints a insertar.

    Returns:
        pandas.DataFrame: DataFrame original con las nuevas filas agregadas.
    """
    for frame, keypoints in enumerate(kp_seq):
        data = {"sample": n_sample, "frame": frame + 1, "keypoints": [keypoints]}
        df_keypoints = pd.DataFrame(data)
        df = pd.concat([df, df_keypoints])

    return df
