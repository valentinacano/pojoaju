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
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint

from app.config import MODEL_FRAMES, MODEL_PATH, MODELS_PATH
from app.database.queries import fetch_word_ids_with_keypoints, fetch_keypoints_for_words, get_word_by_id
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


def _load_sequences(word_ids: list, max_real: int = 60) -> tuple[np.ndarray, np.ndarray]:
    raw = fetch_keypoints_for_words(word_ids)

    grouped = {}
    for word_id, sample_id, frame, kp_json in raw:
        key = (bytes(word_id), sample_id)
        kp = np.array(json.loads(kp_json)) if isinstance(kp_json, str) else np.array(kp_json)
        grouped.setdefault(key, []).append((frame, kp))

    word_to_idx = {bytes(wid): i for i, wid in enumerate(word_ids)}

    sequences, labels = [], []

    # Agrupar por word_id para filtrar las primeras max_real por seña
    by_word = {}
    for (word_id, sample_id) in sorted(grouped.keys()):
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
        return {"error": "Se necesitan al menos 2 palabras con keypoints para entrenar."}

    print(f"📌 {len(word_ids)} palabras encontradas.")
    print("📌 Cargando secuencias...")
    X, y = _load_sequences(word_ids)

    if len(X) == 0:
        return {"error": "No se encontraron secuencias de keypoints."}

    print(f"📌 {len(X)} muestras cargadas. Shape: {X.shape}")

    y_cat = to_categorical(y, num_classes=len(word_ids)).astype(np.float32)

    # Split estratificado 80/20 con seed fija
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_cat,
        test_size=0.2,
        random_state=seed,
        stratify=y
    )

    print(f"📌 Train: {len(X_train)} | Val: {len(X_val)}")

    os.makedirs(MODELS_PATH, exist_ok=True)

    # ✅ Modelo creado DESPUÉS de fijar seeds
    model = get_model(n_classes=len(word_ids))
    model.summary()

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=30,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
    ]

    print("🚀 Entrenando...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=16,
        callbacks=callbacks,
        verbose=2
    )

    epochs_ran = len(history.history["accuracy"])
    final_acc = float(history.history["accuracy"][-1])
    final_val_acc = float(history.history["val_accuracy"][-1])
    final_loss = float(history.history["loss"][-1])
    final_val_loss = float(history.history["val_loss"][-1])

    print(f"\n✅ Entrenamiento finalizado en {epochs_ran} épocas.")
    print(f"   Accuracy:     {final_acc:.4f}")
    print(f"   Val accuracy: {final_val_acc:.4f}")

    return {
        "accuracy":     round(final_acc, 4),
        "val_accuracy": round(final_val_acc, 4),
        "loss":         round(final_loss, 4),
        "val_loss":     round(final_val_loss, 4),
        "epochs_ran":   epochs_ran,
        "n_classes":    len(word_ids),
        "n_samples":    len(X),
    }