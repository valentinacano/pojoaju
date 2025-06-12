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
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=8,
        callbacks=[early_stopping],
    )
    print(model)

    # chekear si esta ok el modelo --------------------
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

    # chekear si esta ok el modelo --------------------

    print("✅ ----- Resumiendo modelo")
    model.summary()
    print(model)

    print("✅ ----- Guardando modelo")
    model.save(MODEL_PATH)
    print(model)
