"""
Tests para ml/keypoints.py
"""

import cv2
import numpy as np
from mediapipe.python.solutions.holistic import Holistic
from ml.keypoints import extract_keypoints, run_mediapipe, has_hand
from app.config import LENGTH_KEYPOINTS


def _dummy_frame():
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


def test_extract_keypoints_shape():
    """El vector de keypoints debe tener exactamente LENGTH_KEYPOINTS valores."""
    with Holistic() as model:
        frame = _dummy_frame()
        results = run_mediapipe(frame, model)
        kp = extract_keypoints(results)
        assert kp.shape == (LENGTH_KEYPOINTS,)


def test_extract_keypoints_sin_deteccion():
    """Frame negro = sin detección = vector de ceros."""
    with Holistic() as model:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = run_mediapipe(frame, model)
        kp = extract_keypoints(results)
        assert kp.shape == (LENGTH_KEYPOINTS,)


def test_has_hand_false_en_frame_vacio():
    with Holistic() as model:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = run_mediapipe(frame, model)
        assert has_hand(results) is False