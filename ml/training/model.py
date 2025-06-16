"""
Definición del modelo LSTM para clasificación de lenguaje de señas.

Este módulo construye una red neuronal secuencial con capas LSTM y Dense
para procesar secuencias de keypoints y predecir la clase correspondiente
a cada muestra capturada.
"""

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.regularizers import l2

from app.config import LENGTH_KEYPOINTS, MODEL_FRAMES


def get_model(output_length: int):
    """
    Construye y compila un modelo LSTM para clasificación multiclase.

    La arquitectura del modelo es:
    - Capa LSTM (64 unidades) con regularización L2 y Dropout.
    - Capa LSTM (128 unidades) con regularización L2 y Dropout.
    - Dos capas Dense ocultas (64 unidades, activación ReLU).
    - Capa de salida Dense con activación softmax.

    Args:
        output_length (int): Número de clases de salida (longitud del vector softmax).

    Returns:
        keras.models.Sequential: Modelo compilado listo para entrenamiento.
    """
    model = Sequential()

    model.add(
        LSTM(
            64,
            return_sequences=True,
            input_shape=(MODEL_FRAMES, LENGTH_KEYPOINTS),
            kernel_regularizer=l2(0.01),
        )
    )
    model.add(Dropout(0.5))
    model.add(LSTM(128, return_sequences=False, kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation="relu", kernel_regularizer=l2(0.001)))
    model.add(Dense(64, activation="relu", kernel_regularizer=l2(0.001)))
    model.add(Dense(output_length, activation="softmax"))
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return model
