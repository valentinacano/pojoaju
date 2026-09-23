"""Partición estratificada y reproducible de muestras reales."""

import numpy as np


def stratified_three_way_indices(
    y: np.ndarray, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reserva por clase train/validación/test sin mezclar el test con el ajuste.

    Cada clase necesita al menos tres muestras: una para cada partición. Para
    clases grandes se reserva aproximadamente 20% para validación y 20% test.
    """
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    train, val, test = [], [], []

    for class_id in np.unique(y):
        indices = np.flatnonzero(y == class_id)
        if len(indices) < 3:
            raise ValueError(
                "Cada palabra necesita al menos 3 muestras reales para separar "
                "entrenamiento, validación y prueba."
            )

        indices = rng.permutation(indices)
        n_test = max(1, int(round(len(indices) * 0.2)))
        n_val = max(1, int(round(len(indices) * 0.2)))
        if n_test + n_val >= len(indices):
            n_test = n_val = 1

        test.extend(indices[:n_test])
        val.extend(indices[n_test : n_test + n_val])
        train.extend(indices[n_test + n_val :])

    return (
        np.asarray(sorted(train), dtype=int),
        np.asarray(sorted(val), dtype=int),
        np.asarray(sorted(test), dtype=int),
    )
