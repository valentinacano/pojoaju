"""
Tests para ml/keypoints.py
"""

import cv2
import numpy as np
from mediapipe.python.solutions.holistic import Holistic
from ml.keypoints import extract_keypoints, run_mediapipe, has_hand, to_model_features
from app.config import LENGTH_KEYPOINTS, RAW_LENGTH_KEYPOINTS


def _dummy_frame():
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


def test_extract_keypoints_shape():
    """El vector crudo debe conservar el formato que se guarda en la BD."""
    with Holistic() as model:
        frame = _dummy_frame()
        results = run_mediapipe(frame, model)
        kp = extract_keypoints(results)
        assert kp.shape == (RAW_LENGTH_KEYPOINTS,)


def test_extract_keypoints_sin_deteccion():
    """Frame negro = sin detección = vector de ceros."""
    with Holistic() as model:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = run_mediapipe(frame, model)
        kp = extract_keypoints(results)
        assert kp.shape == (RAW_LENGTH_KEYPOINTS,)


def test_has_hand_false_en_frame_vacio():
    with Holistic() as model:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = run_mediapipe(frame, model)
        assert has_hand(results) is False


def _synthetic_keypoints(scale=1.0, offset=(0.0, 0.0, 0.0)):
    raw = np.zeros(RAW_LENGTH_KEYPOINTS, dtype=np.float32)
    pose = raw[: 33 * 4].reshape(33, 4)
    base_pose = {
        11: (-0.5, 0.0, 0.0),
        12: (0.5, 0.0, 0.0),
        13: (-0.7, 0.7, 0.1),
        14: (0.7, 0.7, 0.1),
        15: (-0.8, 1.4, 0.2),
        16: (0.8, 1.4, 0.2),
    }
    offset = np.asarray(offset)
    for index, xyz in base_pose.items():
        pose[index, :3] = np.asarray(xyz) * scale + offset
        pose[index, 3] = 1.0

    # Dos formas de mano simples, trasladadas y escaladas junto con la persona.
    for start, wrist in ((1536, (-0.8, 1.4, 0.2)), (1599, (0.8, 1.4, 0.2))):
        hand = raw[start : start + 63].reshape(21, 3)
        wrist = np.asarray(wrist)
        for index in range(21):
            local = np.array([index % 4, index // 4, index * 0.05]) * 0.08
            hand[index] = (wrist + local) * scale + offset
    return raw


def test_model_features_shape():
    features = to_model_features(_synthetic_keypoints())
    assert features.shape == (LENGTH_KEYPOINTS,)


def test_model_features_invariantes_a_distancia_y_posicion():
    reference = to_model_features(_synthetic_keypoints())
    moved = to_model_features(_synthetic_keypoints(scale=2.5, offset=(3.0, -1.5, 0.7)))
    assert np.allclose(reference, moved, atol=1e-5)


def test_mano_ausente_permanece_en_ceros():
    raw = _synthetic_keypoints()
    raw[1536:1599] = 0
    features = to_model_features(raw)
    assert np.all(features[24:87] == 0)
