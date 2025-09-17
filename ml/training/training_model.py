"""
Entrenamiento del modelo de reconocimiento de lenguaje de señas.

... (resto del código) ...
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
    """
    print("✅ ----- Obteniendo words ids")
    word_ids = fetch_word_ids_with_keypoints()
    # Imprime los IDs para verificar si hay datos
    print("IDs de palabras con keypoints:", word_ids)

    print("✅ ----- obteniendio secuencias y etiquetas")
    sequences, labels = get_sequences_and_labels(word_ids)
    # Imprime las secuencias y etiquetas para verificar los datos extraídos
    print("Secuencias:", sequences)
    print("Etiquetas (labels):", labels)

    if not sequences:
        print(
            "❌ Error: No se encontraron secuencias de keypoints. Asegúrate de que los datos de entrenamiento existan en la base de datos."
        )
        return

    sequences = pad_sequences(
        sequences,
        maxlen=int(MODEL_FRAMES),
        padding="pre",
        truncating="post",
        dtype="float16",
    )

    if len(labels) < 2:
        print(
            "❌ Aviso: Solo hay un dato. Duplicando para continuar el entrenamiento de prueba."
        )
        sequences = sequences * 2
        labels = labels * 2

    X = np.array(sequences)
    y = to_categorical(labels).astype(int)

    # El resto del código continúa desde aquí
    early_stopping = EarlyStopping(...)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.05, random_state=42
    )

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
