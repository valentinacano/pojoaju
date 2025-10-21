"""
Tests para la función de entrenamiento del modelo (`training_model`).

Este módulo valida el flujo principal de entrenamiento del modelo LSTM
sin ejecutar realmente el proceso completo de Keras. Se utilizan mocks
para simular la carga de datos, la creación del modelo y el entrenamiento,
verificando que la función devuelva las métricas esperadas y guarde el modelo.

El objetivo es asegurar que la lógica del pipeline funcione correctamente
desde la preparación de datos hasta la devolución del resumen final.
"""

import numpy as np
from unittest.mock import patch, MagicMock
from ml.training.training_model import training_model


@patch("ml.training.training_model.fetch_word_ids_with_keypoints")
@patch("ml.training.training_model.get_sequences_and_labels")
@patch("ml.training.training_model.get_model")
def test_training_model_devuelve_metricas(
    mock_get_model,
    mock_get_sequences_and_labels,
    mock_fetch_word_ids_with_keypoints,
):
    """
    Verifica que `training_model()` retorne métricas correctas simulando un entrenamiento exitoso.

    El test:
    - Mockea la carga de secuencias y etiquetas (`get_sequences_and_labels`).
    - Mockea el modelo (`get_model`) para devolver un objeto falso con un historial de entrenamiento.
    - Simula la existencia de IDs de palabras con keypoints.
    - Valida que la función devuelva un diccionario con las métricas esperadas.

    Returns:
        None: Utiliza aserciones para validar el comportamiento esperado.
    """

    # --- Preparación de mocks ---
    mock_fetch_word_ids = ["id1", "id2", "id3"]
    mock_fetch_word_ids_with_keypoints.return_value = mock_fetch_word_ids

    # Simulamos que existen secuencias y etiquetas
    X_fake = [np.random.rand(15, 1662) for _ in range(3)]
    y_fake = [0, 1, 2]
    mock_get_sequences_and_labels.return_value = (X_fake, y_fake)

    # Mock del modelo con historial simulado
    model_mock = MagicMock()
    model_mock.fit.return_value.history = {
        "accuracy": [0.95],
        "val_accuracy": [0.90],
        "loss": [0.1],
        "val_loss": [0.2],
    }
    model_mock.count_params.return_value = 15000
    model_mock.layers = [1, 2, 3, 4, 5, 6]
    mock_get_model.return_value = model_mock

    # --- Ejecución ---
    result = training_model(epochs=1)

    # --- Validaciones ---
    assert isinstance(result, dict), "El resultado debe ser un diccionario."
    for key in ["accuracy", "val_accuracy", "loss", "val_loss", "params", "layers"]:
        assert key in result, f"Falta la métrica {key} en el resultado."

    assert result["accuracy"] == 0.95
    assert result["val_accuracy"] == 0.9
    assert result["params"] == 15000
    assert result["layers"] == 6
