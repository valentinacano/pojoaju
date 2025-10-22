"""
Entrenamiento optimizado del modelo LSTM para reconocimiento de lenguaje de señas.

Incluye:
- División estratificada del dataset
- Callbacks de EarlyStopping, ModelCheckpoint y ReduceLROnPlateau
- Guardado automático del mejor modelo basado en val_accuracy
- Tamaño de batch ajustado para mayor estabilidad
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

from ml.training.model import get_model
from ml.utils.training_utils import get_sequences_and_labels
from app.database.database_utils import fetch_word_ids_with_keypoints
from app.config import MODEL_FRAMES, MODEL_PATH


def training_model(epochs=700):
    """
    Ejecuta el pipeline completo de entrenamiento del modelo LSTM.

    Args:
        epochs (int): Cantidad máxima de épocas (por defecto 700).

    Returns:
        dict: Métricas finales del modelo.
    """

    print("✅ ----- Obteniendo words ids")
    word_ids = fetch_word_ids_with_keypoints()
    print("IDs de palabras con keypoints:", word_ids)

    print("✅ ----- Obteniendo secuencias y etiquetas")
    sequences, labels = get_sequences_and_labels(word_ids)

    if not sequences:
        print("❌ Error: No se encontraron secuencias de keypoints.")
        return {"error": "No hay datos para entrenar"}

    # --- Preprocesamiento ---
    sequences = pad_sequences(
        sequences,
        maxlen=int(MODEL_FRAMES),
        padding="pre",
        truncating="post",
        dtype="float32",
    )
    X = np.array(sequences)
    y = to_categorical(labels).astype(np.float32)

    # --- Split de datos ---
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=labels
    )

    print(f"✅ Datos listos: {len(X_train)} train | {len(X_val)} val")

    # --- Modelo ---
    print("✅ ----- Creando modelo ajustado")
    model = get_model(len(word_ids))
    model.summary()

    # --- Callbacks ---
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1,
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_accuracy",
        factor=0.5,
        patience=8,
        verbose=1,
        min_lr=1e-5,
        mode="max",
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=12,
        restore_best_weights=True,
        verbose=1,
        mode="max",
    )

    print("✅ ----- Iniciando entrenamiento")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,  # aumentado para estabilidad
        verbose=2,
        callbacks=[checkpoint, reduce_lr, early_stop],
    )

    print("✅ ----- Entrenamiento finalizado")
    print("📦 Mejor modelo guardado en:", MODEL_PATH)

    # --- Métricas finales ---
    final_acc = float(max(history.history["accuracy"]))
    final_val_acc = float(max(history.history["val_accuracy"]))
    final_loss = float(min(history.history["loss"]))
    final_val_loss = float(min(history.history["val_loss"]))

    print("📊 Resultados:")
    print(f"Accuracy final: {final_acc:.4f}")
    print(f"Val Accuracy final: {final_val_acc:.4f}")

    return {
        "accuracy": round(final_acc, 4),
        "val_accuracy": round(final_val_acc, 4),
        "loss": round(final_loss, 4),
        "val_loss": round(final_val_loss, 4),
        "params": model.count_params(),
        "layers": len(model.layers),
    }
