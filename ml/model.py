"""
Arquitectura del modelo LSTM para reconocimiento de señas.
"""

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy

from app.config import MODEL_FRAMES, LENGTH_KEYPOINTS


def get_model(n_classes: int) -> Sequential:
    """
    Construye y compila el modelo LSTM para clasificación multiclase.

    Args:
        n_classes: cantidad de clases (palabras) a clasificar.

    Returns:
        Modelo Keras compilado, listo para entrenar.
    """
    model = Sequential(
        [
            LSTM(
                48,
                return_sequences=True,
                input_shape=(MODEL_FRAMES, LENGTH_KEYPOINTS),
                dropout=0.25,
                recurrent_dropout=0.15,
                kernel_regularizer=l2(0.003),
            ),
            BatchNormalization(),
            Dropout(0.3),
            LSTM(
                64,
                return_sequences=False,
                dropout=0.25,
                recurrent_dropout=0.15,
                kernel_regularizer=l2(0.003),
            ),
            BatchNormalization(),
            Dropout(0.35),
            Dense(64, activation="relu", kernel_regularizer=l2(0.003)),
            BatchNormalization(),
            Dropout(0.35),
            Dense(32, activation="relu", kernel_regularizer=l2(0.003)),
            Dense(n_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss=CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"],
    )

    return model
