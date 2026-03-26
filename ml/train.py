"""
Pipeline completo de entrenamiento del modelo LSTM.

Cambios respecto a la versión anterior:
- test_size corregido: 0.05 → 0.2
- Normalización con normalize_sequence() en vez de pad_sequences()
  (mismo método que se usa en predicción)
- EarlyStopping y ModelCheckpoint agregados
- Estratificación en el split para datasets desbalanceados
"""

import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint

from app.config import MODEL_FRAMES, MODEL_PATH, MODELS_PATH
from app.database.queries import fetch_word_ids_with_keypoints, fetch_keypoints_for_words, get_word_by_id
from ml.model import get_model
from ml.normalize import normalize_sequence


# ---------------------------------------------------------------------------
# Preparación de datos
# ---------------------------------------------------------------------------

def _load_sequences(word_ids: list) -> tuple[np.ndarray, np.ndarray]:
    """
    Carga y prepara las secuencias de keypoints desde la base de datos.

    Para cada muestra:
    1. Agrupa los frames por (word_id, sample_id)
    2. Ordena los frames por número de frame
    3. Normaliza la secuencia a MODEL_FRAMES con normalize_sequence()

    Args:
        word_ids: lista de word_ids a cargar.

    Returns:
        X: np.ndarray de shape (n_samples, MODEL_FRAMES, LENGTH_KEYPOINTS)
        y: np.ndarray de labels enteros (índice de la palabra)
    """
    raw = fetch_keypoints_for_words(word_ids)

    # Agrupar por (word_id, sample_id) → {(word_id, sample_id): [(frame, kp), ...]}
    grouped = {}
    for word_id, sample_id, frame, kp_json in raw:
        key = (bytes(word_id), sample_id)
        kp = np.array(json.loads(kp_json)) if isinstance(kp_json, str) else np.array(kp_json)
        grouped.setdefault(key, []).append((frame, kp))

    word_to_idx = {bytes(wid): i for i, wid in enumerate(word_ids)}

    sequences, labels = [], []
    for (word_id, _), frames in grouped.items():
        ordered = [kp for _, kp in sorted(frames, key=lambda x: x[0])]
        normalized = normalize_sequence(ordered, MODEL_FRAMES)  # ← mismo método que predicción
        sequences.append(normalized)
        labels.append(word_to_idx[word_id])

    return np.array(sequences, dtype=np.float32), np.array(labels)


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------

def train(epochs: int = 300) -> dict:
    """
    Ejecuta el pipeline completo de entrenamiento.

    Pasos:
        1. Carga word_ids con keypoints desde la base de datos
        2. Carga y normaliza las secuencias
        3. Split train/val 80-20 estratificado
        4. Entrena con EarlyStopping (patience=30)
        5. Guarda el mejor modelo automáticamente
        6. Retorna métricas finales

    Args:
        epochs: máximo de épocas (EarlyStopping puede cortar antes).

    Returns:
        dict con accuracy, val_accuracy, loss, val_loss, epochs_ran, n_classes.
    """
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

    # Split estratificado 80/20
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_cat,
        test_size=0.2,
        random_state=42,
        stratify=y  # garantiza representación de todas las clases
    )

    print(f"📌 Train: {len(X_train)} | Val: {len(X_val)}")

    os.makedirs(MODELS_PATH, exist_ok=True)
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