"""
Definición del modelo LSTM optimizado para clasificación de lenguaje de señas.

Este modelo está ajustado para mejorar la precisión sin caer en sobreajuste.
Se incrementó ligeramente la cantidad de neuronas y se redujo el Dropout,
manteniendo la normalización y la regularización L2.

Arquitectura del modelo:
- LSTM (48 unidades, recurrent_dropout=0.25, regularización L2)
- LayerNormalization
- Dropout (0.3)
- LSTM (96 unidades, recurrent_dropout=0.25, regularización L2)
- LayerNormalization
- Dropout (0.3)
- Dense (48 unidades, ReLU)
- Dropout (0.2)
- Capa de salida Dense (Softmax)
"""

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, LayerNormalization
from keras.regularizers import l2
from keras.losses import CategoricalCrossentropy
from keras.optimizers import AdamW

from app.config import LENGTH_KEYPOINTS, MODEL_FRAMES


def get_model(output_length: int):
    """
    Construye y compila un modelo LSTM ajustado para clasificación multiclase.

    Args:
        output_length (int): Número de clases de salida (longitud del vector softmax).

    Returns:
        keras.models.Sequential: Modelo compilado listo para entrenamiento.
    """
    model = Sequential()

    # Primera capa LSTM
    model.add(
        LSTM(
            48,
            return_sequences=True,
            recurrent_dropout=0.25,
            input_shape=(MODEL_FRAMES, LENGTH_KEYPOINTS),
            kernel_regularizer=l2(1e-4),
        )
    )
    model.add(LayerNormalization())
    model.add(Dropout(0.3))

    # Segunda capa LSTM
    model.add(
        LSTM(
            96,
            return_sequences=False,
            recurrent_dropout=0.25,
            kernel_regularizer=l2(1e-4),
        )
    )
    model.add(LayerNormalization())
    model.add(Dropout(0.3))

    # Capa intermedia
    model.add(Dense(48, activation="relu", kernel_regularizer=l2(1e-4)))
    model.add(Dropout(0.2))

    # Capa de salida
    model.add(Dense(output_length, activation="softmax"))

    # Compilación del modelo
    optimizer = AdamW(learning_rate=1e-3, weight_decay=1e-4)
    loss = CategoricalCrossentropy(label_smoothing=0.1)
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

    return model
