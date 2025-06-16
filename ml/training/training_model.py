"""
Entrenamiento del modelo de reconocimiento de lenguaje de señas.

Este módulo recupera los datos de entrenamiento desde la base de datos,
preprocesa las secuencias de keypoints y entrena un modelo de clasificación
usando una arquitectura LSTM.

El modelo entrenado se guarda en disco para su uso posterior.
"""

import numpy as np

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

from ml.training.model import get_model
from ml.utils.training_utils import get_sequences_and_labels
from app.database.database_utils import fetch_word_ids_with_keypoints
from app.config import MODEL_FRAMES, MODEL_PATH


def training_model(epochs=500):
    """
    Entrena un modelo LSTM de clasificación multiclase a partir de secuencias de keypoints.

    Este pipeline realiza las siguientes etapas:
    - Recupera los `word_ids` que tienen keypoints en la base.
    - Obtiene y preprocesa las secuencias y etiquetas.
    - Divide los datos en entrenamiento y validación.
    - Entrena un modelo LSTM usando Keras.
    - Guarda el modelo entrenado en el disco.

    Args:
        epochs (int, opcional): Número de épocas de entrenamiento (por defecto 500).

    Returns:
        None: Esta función no retorna nada. Guarda el modelo entrenado en la ruta definida.
    """
    print("✅ ----- Obteniendo words ids")
    word_ids = fetch_word_ids_with_keypoints()

    print("✅ ----- obteniendio secuencias y etiquetas")
    sequences, labels = get_sequences_and_labels(word_ids)
    print(labels)

    sequences = pad_sequences(
        sequences,
        maxlen=int(MODEL_FRAMES),
        padding="pre",
        truncating="post",
        dtype="float16",
    )

    X = np.array(sequences)
    y = to_categorical(labels).astype(int)

    early_stopping = EarlyStopping(
        monitor="accuracy", patience=10, restore_best_weights=True
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.05, random_state=42
    )

    print("✅ ----- Obteniendo modelo")
    model = get_model(len(word_ids))
    print(model)

    print("✅ ----- Enrenando modelo")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=8,
        callbacks=[early_stopping],
    )

    print("Historial de entrenamiento:")
    print("Accuracy final:", history.history["accuracy"][-1])
    print("Val Accuracy final:", history.history["val_accuracy"][-1])

    print("✅ ----- Resumiendo modelo")
    model.summary()

    print("✅ ----- Guardando modelo")
    model.save(MODEL_PATH)
