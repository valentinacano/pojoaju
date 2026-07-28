"""
Evaluación del modelo — matriz de confusión y métricas.

Correcciones respecto a la versión anterior:
- Labels muestran nombres reales de palabras (no direcciones de memoria)
- test_size corregido: 0.8 → 0.3 (el 0.8 anterior era un bug — entrenaba con el 20%)
- Usa normalize_sequence() igual que train.py y predict.py
"""

import os
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)
from keras.models import load_model

from app.config import MODEL_PATH, MODEL_FRAMES
from app.database.queries import (
    fetch_word_ids_with_keypoints,
    fetch_keypoints_for_words,
    get_word_by_id,
)
from ml.normalize import normalize_sequence


CONFUSION_PATH = "app/views/static/confusion/confusion_matrix.png"


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


def generate_confusion_matrix(save_path: str = CONFUSION_PATH) -> tuple:
    """
    Genera y guarda la matriz de confusión del modelo actual.

    Returns:
        tuple: (cm, y_val, y_pred, metrics_dict)
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    word_ids = fetch_word_ids_with_keypoints()

    if len(word_ids) < 2:
        raise ValueError("Se necesitan al menos 2 palabras con keypoints.")

    # Construir mapeo índice → nombre real
    idx_to_word = {}
    for i, wid in enumerate(word_ids):
        row = get_word_by_id(bytes(wid))
        idx_to_word[i] = row[1] if row else f"clase_{i}"  # row[1] = word string

    X, y = _load_sequences(word_ids)

    val_size = max(0.2, len(word_ids) / len(X))
    val_size = min(0.5, val_size)

    _, X_val, _, y_val = train_test_split(
        X,
        y,
        test_size=val_size,
        random_state=42,
        stratify=y,
    )

    model = load_model(MODEL_PATH)
    y_pred_probs = model.predict(X_val, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    cm = confusion_matrix(y_val, y_pred)
    unique_classes = sorted(set(y_val))
    labels_text = [idx_to_word[i] for i in unique_classes]

    accuracy = float(np.sum(y_val == y_pred) / len(y_val))
    report = classification_report(
        y_val, y_pred, target_names=labels_text, output_dict=True, zero_division=0
    )

    # Graficar
    n = len(unique_classes)
    fig, ax = plt.subplots(figsize=(max(10, n * 0.6), max(8, n * 0.5)))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_text).plot(
        ax=ax, xticks_rotation=45, cmap="Blues", colorbar=True
    )
    plt.title(
        f"Matriz de Confusión — Pojoaju\nAccuracy: {accuracy:.2%}", fontsize=14, pad=20
    )
    plt.xlabel("Predicción", fontsize=12)
    plt.ylabel("Real", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"✅ Matriz guardada en: {save_path}")
    print(f"✅ Accuracy: {accuracy:.2%}")

    metrics = {
        "accuracy": accuracy,
        "report": report,
        "n_classes": n,
        "n_samples": len(X_val),
        "labels": labels_text,
    }

    return cm, y_val, y_pred, metrics


def get_top_confusions(cm: np.ndarray, labels: list, top_k: int = 5) -> list:
    """
    Retorna los top K pares de palabras más confundidas.

    Args:
        cm: matriz de confusión.
        labels: nombres de las clases.
        top_k: cantidad de confusiones a retornar.

    Returns:
        lista de (palabra_real, palabra_predicha, cantidad)
    """
    confusions = [
        (labels[i], labels[j], int(cm[i, j]))
        for i in range(len(cm))
        for j in range(len(cm))
        if i != j and cm[i, j] > 0
    ]
    return sorted(confusions, key=lambda x: x[2], reverse=True)[:top_k]
