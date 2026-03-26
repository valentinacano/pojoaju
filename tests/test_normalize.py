"""
Tests para ml/normalize.py

Verifica que normalize_sequence sea idéntica en todos los contextos.
"""

import numpy as np
import pytest
from ml.normalize import normalize_sequence
from app.config import MODEL_FRAMES


def test_normalize_longitud_correcta():
    seq = [np.random.rand(1662) for _ in range(8)]
    result = normalize_sequence(seq)
    assert result.shape == (MODEL_FRAMES, 1662)


def test_normalize_secuencia_larga():
    seq = [np.random.rand(1662) for _ in range(30)]
    result = normalize_sequence(seq)
    assert result.shape == (MODEL_FRAMES, 1662)


def test_normalize_secuencia_exacta():
    seq = [np.random.rand(1662) for _ in range(MODEL_FRAMES)]
    result = normalize_sequence(seq)
    assert result.shape == (MODEL_FRAMES, 1662)
    assert np.allclose(result, np.array(seq))


def test_normalize_consistencia_training_prediccion():
    """
    La misma secuencia debe producir el mismo resultado
    sin importar cuántas veces se llame.
    """
    seq = [np.random.rand(1662) for _ in range(10)]
    r1 = normalize_sequence(seq)
    r2 = normalize_sequence(seq)
    assert np.allclose(r1, r2)