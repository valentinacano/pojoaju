"""
Entrenamiento del modelo de reconocimiento de lenguaje de señas.

Este módulo carga los keypoints almacenados en la base de datos, prepara los datos
(ajustando secuencias y etiquetas), entrena un modelo LSTM y guarda el modelo final.

Incluye:
- Carga y preprocesamiento de los datos (padded sequences, one-hot labels)
- División en training y validation sets
- Entrenamiento del modelo LSTM definido en `ml.training.model`
- Guardado del modelo en `MODEL_PATH`
- Retorno de métricas finales para visualización en interfaz web

Funciones:
- training_model(epochs=500): ejecuta todo el pipeline de entrenamiento y retorna métricas clave.
"""


import numpy as np

from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

from ml.training.model import get_model
from ml.utils.training_utils import get_sequences_and_labels
from app.database.database_utils import fetch_word_ids_with_keypoints
from app.config import MODEL_FRAMES, MODEL_PATH


def training_model(epochs=500):
    """
    Ejecuta el pipeline completo de entrenamiento del modelo LSTM.

    Incluye la carga de datos, preparación de secuencias, entrenamiento
    y guardado del modelo. Retorna un resumen con métricas finales.

    Args:
        epochs (int): Cantidad de épocas de entrenamiento (por defecto 500).

    Returns:
        dict: Diccionario con métricas finales: accuracy, val_accuracy, loss, val_loss, etc.
    """

    print("✅ ----- Obteniendo words ids")
    word_ids = fetch_word_ids_with_keypoints()
    print("IDs de palabras con keypoints:", word_ids)

    print("✅ ----- obteniendo secuencias y etiquetas")
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
        dtype="float16",
    )
    X = np.array(sequences)
    y = to_categorical(labels).astype(int)

    # --- Split ---
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.05, random_state=42
    )

    print("✅ ----- Obteniendo modelo")
    model = get_model(len(word_ids))
    print(model)

    print("✅ ----- Entrenando modelo")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=8,
        verbose=2,
    )

    # Guardamos métricas
    final_acc = float(history.history["accuracy"][-1])
    final_val_acc = float(history.history["val_accuracy"][-1])
    final_loss = float(history.history["loss"][-1])
    final_val_loss = float(history.history["val_loss"][-1])

    print("Historial de entrenamiento:")
    print("Accuracy final:", final_acc)
    print("Val Accuracy final:", final_val_acc)

    print("✅ ----- Resumiendo modelo")
    model.summary()

    print("✅ ----- Guardando modelo")
    model.save(MODEL_PATH)

    # 👇 Retornamos un diccionario solo con lo útil para el HTML
    return {
        "accuracy": round(final_acc, 4),
        "val_accuracy": round(final_val_acc, 4),
        "loss": round(final_loss, 4),
        "val_loss": round(final_val_loss, 4),
        "params": model.count_params(),  # total parámetros entrenables
        "layers": len(model.layers),  # cantidad de capas
    }
