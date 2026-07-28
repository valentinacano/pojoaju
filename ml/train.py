"""
Pipeline completo de entrenamiento del modelo LSTM.

Cambios respecto a la versión anterior:
- test_size corregido: 0.05 → 0.2
- Normalización con normalize_sequence() en vez de pad_sequences()
- EarlyStopping y ModelCheckpoint
- Estratificación en el split
- Orden determinístico de muestras para resultados reproducibles
- Seeds globales fijas para reproducibilidad total
"""

import os
import random
import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from app.config import MODEL_FRAMES, MODEL_PATH, MODELS_PATH
from app.database.queries import (
    fetch_word_ids_with_keypoints,
    fetch_keypoints_for_words,
    get_word_by_id,
)
from ml.model import get_model
from ml.normalize import normalize_sequence


def set_seeds(seed: int = 42):
    """
    Fija todas las seeds globales para garantizar reproducibilidad total.
    Debe llamarse ANTES de crear el modelo y ANTES del split.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _load_sequences(
    word_ids: list, max_real: int = 60
) -> tuple[np.ndarray, np.ndarray]:
    raw = fetch_keypoints_for_words(word_ids)

    grouped = {}
    for word_id, sample_id, frame, kp_json in raw:
        key = (bytes(word_id), sample_id)
        kp = (
            np.array(json.loads(kp_json))
            if isinstance(kp_json, str)
            else np.array(kp_json)
        )
        grouped.setdefault(key, []).append((frame, kp))

    word_to_idx = {bytes(wid): i for i, wid in enumerate(word_ids)}

    sequences, labels = [], []

    # Agrupar por word_id para filtrar las primeras max_real por seña
    by_word = {}
    for word_id, sample_id in sorted(grouped.keys()):
        by_word.setdefault(word_id, []).append(sample_id)

    for word_id, sample_ids in by_word.items():
        # ✅ Solo las primeras max_real muestras por seña
        for sample_id in sample_ids[:max_real]:
            frames = grouped[(word_id, sample_id)]
            ordered = [kp for _, kp in sorted(frames, key=lambda x: x[0])]
            normalized = normalize_sequence(ordered, MODEL_FRAMES)
            sequences.append(normalized)
            labels.append(word_to_idx[word_id])

    return np.array(sequences, dtype=np.float32), np.array(labels)


def _augment_sequences(
    X: np.ndarray, y: np.ndarray, factor: int = 2, noise_std: float = 0.01
) -> tuple[np.ndarray, np.ndarray]:
    """
    Genera variantes suaves solo para el set de entrenamiento.

    No se guardan en la BD y nunca entran al set de validación, así la métrica
    sigue midiendo muestras reales.
    """
    if factor <= 0:
        return X, y

    augmented_X = [X]
    augmented_y = [y]

    for _ in range(factor):
        warped = X.copy()

        # Ruido leve en landmarks para tolerar pequeñas variaciones de postura.
        warped += np.random.normal(0, noise_std, size=warped.shape).astype(np.float32)

        # Desplazamiento temporal de hasta 2 frames, con padding por borde.
        for i in range(len(warped)):
            shift = np.random.randint(-2, 3)
            if shift == 0:
                continue
            warped[i] = np.roll(warped[i], shift=shift, axis=0)
            if shift > 0:
                warped[i, :shift] = warped[i, shift]
            else:
                warped[i, shift:] = warped[i, shift - 1]

        augmented_X.append(warped.astype(np.float32))
        augmented_y.append(y)

    return np.concatenate(augmented_X), np.concatenate(augmented_y)


def _class_distribution(y: np.ndarray) -> dict:
    counts = np.bincount(y)
    nonzero = counts[counts > 0]
    return {
        "min": int(nonzero.min()),
        "max": int(nonzero.max()),
        "mean": round(float(nonzero.mean()), 2),
    }


def train(epochs: int = 300, seed: int = 42) -> dict:
    """
    Ejecuta el pipeline completo de entrenamiento.

    Args:
        epochs: máximo de épocas (EarlyStopping puede cortar antes).
        seed: semilla global para reproducibilidad.

    Returns:
        dict con accuracy, val_accuracy, loss, val_loss, epochs_ran, n_classes.
    """

    # ✅ Seeds fijas ANTES de todo
    set_seeds(seed)

    print("📌 Cargando word_ids...")
    word_ids = fetch_word_ids_with_keypoints()

    if len(word_ids) < 2:
        return {
            "error": "Se necesitan al menos 2 palabras con keypoints para entrenar."
        }

    print(f"📌 {len(word_ids)} palabras encontradas.")
    print("📌 Cargando secuencias...")
    X, y = _load_sequences(word_ids)

    if len(X) == 0:
        return {"error": "No se encontraron secuencias de keypoints."}

    print(f"📌 {len(X)} muestras cargadas. Shape: {X.shape}")

    distribution = _class_distribution(y)

    if distribution["min"] < 2:
        return {
            "error": (
                "Cada palabra necesita al menos 2 muestras para hacer validación "
                "estratificada. Agregá más muestras reales a las clases con 1 sola muestra."
            )
        }

    val_size = max(0.2, len(word_ids) / len(X))
    val_size = min(0.5, val_size)

    # Split estratificado con seed fija. La validación debe tener al menos
    # una muestra por clase para que la métrica no ignore palabras.
    X_train, X_val, y_train_idx, y_val_idx = train_test_split(
        X, y, test_size=val_size, random_state=seed, stratify=y
    )

    print(f"📌 Train: {len(X_train)} | Val: {len(X_val)}")

    X_train, y_train_idx = _augment_sequences(X_train, y_train_idx)
    y_train = to_categorical(y_train_idx, num_classes=len(word_ids)).astype(np.float32)
    y_val = to_categorical(y_val_idx, num_classes=len(word_ids)).astype(np.float32)

    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.arange(len(word_ids)), y=y_train_idx
    )
    class_weight = {i: float(weight) for i, weight in enumerate(class_weights)}

    print(f"📌 Train aumentado: {len(X_train)} muestras.")
    print(
        "📌 Muestras por clase "
        f"(min/prom/max): {distribution['min']}/{distribution['mean']}/{distribution['max']}"
    )

    os.makedirs(MODELS_PATH, exist_ok=True)

    # ✅ Modelo creado DESPUÉS de fijar seeds
    model = get_model(n_classes=len(word_ids))
    model.summary()

    callbacks = [
        EarlyStopping(
            monitor="val_loss", patience=40, restore_best_weights=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=10, min_lr=0.00005, verbose=1
        ),
        ModelCheckpoint(
            filepath=MODEL_PATH, monitor="val_loss", save_best_only=True, verbose=1
        ),
    ]

    print("🚀 Entrenando...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=16,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=2,
    )

    epochs_ran = len(history.history["accuracy"])
    best_epoch_idx = int(np.argmin(history.history["val_loss"]))
    final_acc = float(history.history["accuracy"][best_epoch_idx])
    final_val_acc = float(history.history["val_accuracy"][best_epoch_idx])
    final_loss = float(history.history["loss"][best_epoch_idx])
    final_val_loss = float(history.history["val_loss"][best_epoch_idx])

    print(f"\n✅ Entrenamiento finalizado en {epochs_ran} épocas.")
    print(f"   Mejor época:  {best_epoch_idx + 1}")
    print(f"   Accuracy:     {final_acc:.4f}")
    print(f"   Val accuracy: {final_val_acc:.4f}")

    return {
        "accuracy": round(final_acc, 4),
        "val_accuracy": round(final_val_acc, 4),
        "loss": round(final_loss, 4),
        "val_loss": round(final_val_loss, 4),
        "epochs_ran": epochs_ran,
        "best_epoch": best_epoch_idx + 1,
        "n_classes": len(word_ids),
        "n_samples": len(X),
        "class_min_samples": distribution["min"],
        "class_max_samples": distribution["max"],
    }
