"""
Arquitectura del modelo LSTM para reconocimiento de señas.

Cambios respecto a la versión anterior:
- Dropout reducido de 0.5 → 0.2 (era demasiado agresivo para datasets pequeños)
- L2 uniforme en todas las capas (antes era 10x más fuerte en la primera)
- BatchNormalization agregado para estabilizar el entrenamiento
- Función get_model() recibe output_length dinámicamente
"""

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.regularizers import l2

from app.config import MODEL_FRAMES, LENGTH_KEYPOINTS


def get_model(n_classes: int) -> Sequential:
    """
    Construye y compila el modelo LSTM para clasificación multiclase.

    Arquitectura:
        LSTM(64)  → BatchNorm → Dropout(0.2)
        LSTM(128) → BatchNorm → Dropout(0.2)
        Dense(64, relu) → Dropout(0.2)
        Dense(64, relu)
        Dense(n_classes, softmax)

    Args:
        n_classes: cantidad de clases (palabras) a clasificar.

    Returns:
        Modelo Keras compilado, listo para entrenar.
    """
    model = Sequential([
        LSTM(64,
             return_sequences=True,
             input_shape=(MODEL_FRAMES, LENGTH_KEYPOINTS),
             kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.2),

        LSTM(128,
             return_sequences=False,
             kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.2),

        Dense(64, activation="relu", kernel_regularizer=l2(0.001)),
        Dropout(0.2),
        Dense(64, activation="relu"),

        Dense(n_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model