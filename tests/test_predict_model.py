"""
Tests para la función de predicción desde cámara (`predict_model_from_camera`).

Este módulo valida el flujo completo de predicción utilizando mocks para simular
la cámara, el modelo LSTM y los resultados de MediaPipe. Se comprueba que el sistema
detecta correctamente la transición de “mano presente” a “mano ausente” y genera una
predicción con texto audible.

No se requiere hardware real ni modelo entrenado: todas las dependencias se mockean
para aislar la lógica principal.
"""

import numpy as np
from unittest.mock import patch, MagicMock

from ml.prediction.predict_model_from_camera import predict_model_from_camera


@patch("ml.prediction.predict_model_from_camera.draw_keypoints")
@patch("ml.prediction.predict_model_from_camera.text_to_speech")
@patch("ml.prediction.predict_model_from_camera.there_hand")
@patch("ml.prediction.predict_model_from_camera.fetch_word_ids_with_keypoints")
@patch("ml.prediction.predict_model_from_camera.search_word_id")
@patch("ml.prediction.predict_model_from_camera.mediapipe_detection")
@patch("ml.prediction.predict_model_from_camera.extract_keypoints")
@patch("ml.prediction.predict_model_from_camera.Holistic")
@patch("ml.prediction.predict_model_from_camera.load_model")
def test_predict_model_desde_video(
    mock_load_model,
    mock_Holistic,
    mock_extract_keypoints,
    mock_detection,
    mock_search_word,
    mock_fetch_ids,
    mock_there_hand,
    mock_tts,
    mock_draw,
):
    """
    Verifica que `predict_model_from_camera()` procese frames simulados y genere una predicción válida.

    El test:
    - Mockea el modelo LSTM (`load_model`) para que siempre prediga "hola".
    - Simula 15 frames con detección de mano y 5 sin mano.
    - Mockea la cámara (`cv2.VideoCapture`) para devolver un número fijo de frames.
    - Desactiva la visualización (`cv2.imshow`, `cv2.waitKey`).
    - Valida que la salida sea una lista con al menos una predicción que contenga "hola".

    Returns:
        None: Utiliza aserciones para validar el comportamiento esperado.
    """

    # --- Preparación de mocks ---
    mock_fetch_ids.return_value = ["id1"]
    mock_search_word.return_value = ("id1", "hola", "saludo")

    # Modelo falso: siempre devuelve una predicción de alta confianza
    model_fake = MagicMock()
    model_fake.predict.return_value = np.array([[0.99]])
    mock_load_model.return_value = model_fake

    # Resultados de MediaPipe (con mano presente)
    results_fake = MagicMock()
    results_fake.left_hand_landmarks = True
    mock_detection.return_value = results_fake

    # Secuencia simulada de detección de mano
    mock_there_hand.side_effect = [True] * 15 + [False] * 5
    mock_extract_keypoints.side_effect = [np.random.rand(1662) for _ in range(15)]

    # Mock del modelo Holistic y de VideoCapture
    mock_Holistic.return_value.__enter__.return_value = MagicMock()
    mock_video = MagicMock()
    mock_video.read.side_effect = [
        (True, np.zeros((480, 640, 3), dtype=np.uint8))
    ] * 20 + [(False, None)]
    mock_capture = MagicMock(return_value=mock_video)

    # --- Ejecución ---
    with patch("cv2.VideoCapture", mock_capture), patch("cv2.imshow"), patch(
        "cv2.waitKey", return_value=-1
    ), patch("cv2.destroyAllWindows"):
        result = predict_model_from_camera(threshold=0.5)

    # --- Validaciones ---
    assert isinstance(result, list), "El resultado debe ser una lista."
    assert any(
        "hola" in s for s in result
    ), f"No se encontró 'hola' en el resultado: {result}"
