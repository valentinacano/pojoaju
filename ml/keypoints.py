"""
Extracción y preparación de keypoints con MediaPipe Holistic.

Este módulo es usado tanto en captura como en predicción.
Los 1662 valores crudos se conservan en la BD. Antes de entrar al modelo se
convierten en 150 features centradas en brazos y manos.
"""

import os
import cv2
import numpy as np
from mediapipe.python.solutions.holistic import Holistic
from app.config import LENGTH_KEYPOINTS, RAW_LENGTH_KEYPOINTS


POSE_VALUES = 33 * 4
FACE_VALUES = 468 * 3
HAND_VALUES = 21 * 3
LEFT_HAND_START = POSE_VALUES + FACE_VALUES
RIGHT_HAND_START = LEFT_HAND_START + HAND_VALUES

# Hombros, codos y muñecas de MediaPipe Pose.
ARM_LANDMARKS = (11, 12, 13, 14, 15, 16)


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
        np.array(
            [
                [lm.x, lm.y, lm.z, lm.visibility]
                for lm in results.pose_landmarks.landmark
            ]
        ).flatten()
        if results.pose_landmarks
        else np.zeros(33 * 4)
    )
    face = (
        np.array(
            [[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark]
        ).flatten()
        if results.face_landmarks
        else np.zeros(468 * 3)
    )
    lh = (
        np.array(
            [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]
        ).flatten()
        if results.left_hand_landmarks
        else np.zeros(21 * 3)
    )
    rh = (
        np.array(
            [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]
        ).flatten()
        if results.right_hand_landmarks
        else np.zeros(21 * 3)
    )
    return np.concatenate([pose, face, lh, rh])


def _normalize_hand(hand: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Centra una mano en la muñeca y la escala por el tamaño de la palma.

    Esto conserva la forma y orientación de los dedos, pero elimina el efecto
    de acercar la mano a la cámara. Una mano ausente permanece en ceros.
    """
    if not np.any(hand):
        return np.zeros(HAND_VALUES, dtype=np.float32)

    centered = hand - hand[0]
    # Distancia media desde la muñeca a los MCP: índice, medio, anular y meñique.
    palm_scale = np.mean(
        [np.linalg.norm(centered[index, :2]) for index in (5, 9, 13, 17)]
    )
    if palm_scale < eps:
        return np.zeros(HAND_VALUES, dtype=np.float32)

    return (centered / palm_scale).astype(np.float32).flatten()


def to_model_features(keypoints: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Convierte los 1662 keypoints crudos en las 150 features del modelo.

    - Elimina cara, cabeza, cadera y piernas.
    - Centra los brazos entre ambos hombros.
    - Escala los brazos por el ancho de hombros.
    - Centra y escala cada mano de forma independiente.

    La normalización reduce la dependencia de la posición de la persona y de
    su distancia a la cámara. El método acepta features ya transformadas para
    facilitar su uso seguro en los distintos pipelines.
    """
    raw = np.asarray(keypoints, dtype=np.float32).reshape(-1)

    if raw.size == LENGTH_KEYPOINTS:
        return raw
    if raw.size != RAW_LENGTH_KEYPOINTS:
        raise ValueError(
            f"Se esperaban {RAW_LENGTH_KEYPOINTS} keypoints crudos o "
            f"{LENGTH_KEYPOINTS} features, se recibieron {raw.size}."
        )

    pose = raw[:POSE_VALUES].reshape(33, 4)
    arms = pose[list(ARM_LANDMARKS)].copy()

    shoulders = pose[[11, 12], :3]
    shoulder_center = shoulders.mean(axis=0)
    shoulder_scale = np.linalg.norm(shoulders[0, :2] - shoulders[1, :2])

    if shoulder_scale >= eps:
        arms[:, :3] = (arms[:, :3] - shoulder_center) / shoulder_scale
    else:
        arms[:] = 0

    left_hand = raw[LEFT_HAND_START:RIGHT_HAND_START].reshape(21, 3)
    right_hand = raw[RIGHT_HAND_START:].reshape(21, 3)

    features = np.concatenate(
        [
            arms.flatten(),
            _normalize_hand(left_hand, eps),
            _normalize_hand(right_hand, eps),
        ]
    ).astype(np.float32)

    if features.size != LENGTH_KEYPOINTS:
        raise RuntimeError(
            f"Vector de modelo inválido: {features.size} != {LENGTH_KEYPOINTS}"
        )
    return features


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
        key=lambda x: int(os.path.splitext(x)[0].split("_")[-1]),
    )
    sequence = []
    for fname in files:
        frame = cv2.imread(os.path.join(folder_path, fname))
        if frame is not None:
            results = run_mediapipe(frame, model)
            sequence.append(extract_keypoints(results))

    return np.array(sequence)
