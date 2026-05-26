"""
Animador de señas usando keypoints almacenados en la BD.

Mejoras:
1. Usa la muestra más representativa (más cercana al centroide)
2. Interpola frames intermedios para suavizar la animación
3. Maneja muestras con distinto número de frames
"""

import json
import numpy as np
from app.database.queries import (
    fetch_keypoints_for_words,
    get_word_by_id,
    word_to_id,
    fetch_word_ids_with_keypoints,
)

POSE_START, POSE_END = 0, 132
LH_START, LH_END = 1536, 1599
RH_START, RH_END = 1599, 1662

POSE_CONNECTIONS = [
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 23),
    (12, 24),
    (23, 24),
]

HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (5, 9),
    (9, 13),
    (13, 17),
]


def _parse_keypoints(kp_json) -> np.ndarray:
    if isinstance(kp_json, str):
        return np.array(json.loads(kp_json))
    return np.array(kp_json)


def _extract_pose(kp: np.ndarray) -> list[dict]:
    pose = kp[POSE_START:POSE_END].reshape(33, 4)
    return [{"x": float(p[0]), "y": float(p[1])} for p in pose]


def _extract_hand(kp: np.ndarray, start: int, end: int) -> list[dict]:
    hand = kp[start:end].reshape(21, 3)
    return [{"x": float(h[0]), "y": float(h[1])} for h in hand]


def _get_best_sample(grouped: dict) -> list[np.ndarray]:
    """
    Encuentra la muestra más representativa (más cercana al centroide).
    Normaliza todas las muestras al mismo tamaño antes de comparar.
    """
    if len(grouped) == 1:
        sid = list(grouped.keys())[0]
        frames = grouped[sid]
        return [frames[f] for f in sorted(frames.keys())]

    max_len = max(len(frames) for frames in grouped.values())

    sample_vectors = {}
    for sample_id, frames in grouped.items():
        ordered = [frames[f] for f in sorted(frames.keys())]
        while len(ordered) < max_len:
            ordered.append(ordered[-1])
        sample_vectors[sample_id] = np.concatenate(ordered[:max_len])

    matrix = np.stack(list(sample_vectors.values()))
    centroid = np.mean(matrix, axis=0)

    best_id = min(
        sample_vectors.keys(),
        key=lambda sid: np.linalg.norm(sample_vectors[sid] - centroid),
    )

    frames = grouped[best_id]
    return [frames[f] for f in sorted(frames.keys())]


def _interpolate_frames(frames: list[np.ndarray], factor: int = 3) -> list[np.ndarray]:
    """Interpola frames intermedios para suavizar la animación."""
    if len(frames) < 2:
        return frames
    result = []
    for i in range(len(frames) - 1):
        result.append(frames[i])
        for j in range(1, factor):
            t = j / factor
            result.append((1 - t) * frames[i] + t * frames[i + 1])
    result.append(frames[-1])
    return result


def get_sign_animation(word: str, interpolation_factor: int = 3) -> dict | None:
    """Genera la animación de una seña usando la mejor muestra interpolada."""
    wid = word_to_id(word)
    raw = fetch_keypoints_for_words([wid])
    if not raw:
        return None

    grouped = {}
    for _, sample_id, frame, kp_json in raw:
        kp = _parse_keypoints(kp_json)
        grouped.setdefault(sample_id, {})[frame] = kp

    best_frames = _get_best_sample(grouped)
    smooth_frames = _interpolate_frames(best_frames, factor=interpolation_factor)

    animation_frames = []
    for kp in smooth_frames:
        animation_frames.append(
            {
                "pose": _extract_pose(kp),
                "left_hand": _extract_hand(kp, LH_START, LH_END),
                "right_hand": _extract_hand(kp, RH_START, RH_END),
            }
        )

    return {
        "word": word,
        "frames": animation_frames,
        "pose_connections": POSE_CONNECTIONS,
        "hand_connections": HAND_CONNECTIONS,
    }


def get_available_words() -> list[str]:
    """Retorna las palabras que tienen keypoints en la BD."""
    word_ids = fetch_word_ids_with_keypoints()
    words = []
    for wid in word_ids:
        row = get_word_by_id(bytes(wid))
        if row:
            words.append(row[1])
    return sorted(words)
