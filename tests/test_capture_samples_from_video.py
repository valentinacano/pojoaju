"""
Tests para la función `capture_samples_from_video`.

Este módulo crea un video artificial con contenido simulado (una figura negra como mano)
y verifica que la función capture correctamente una secuencia de muestras desde el video.
"""

import os
import cv2
import numpy as np
import pytest
from ml.features.capture_samples_video import capture_samples_from_video
from unittest.mock import patch, MagicMock


@pytest.fixture
def dummy_video(tmp_path):
    """
    Crea un video temporal con frames artificiales para testear la función de captura.

    Genera un archivo `.avi` con 30 frames blancos y un rectángulo negro que simula
    una mano en movimiento. También crea una carpeta de salida vacía para guardar las muestras.

    Args:
        tmp_path (Path): Carpeta temporal generada por pytest.

    Returns:
        tuple[str, str]: Ruta al archivo de video generado y ruta de la carpeta de salida.
    """
    video_path = str(tmp_path / "dummy_video.avi")
    output_path = str(tmp_path / "output_samples")
    os.makedirs(output_path, exist_ok=True)

    height, width = 480, 640
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(video_path, fourcc, 10.0, (width, height))

    # Crear 30 frames con una mano dibujada (simulada)
    for _ in range(30):
        frame = np.full((height, width, 3), 255, dtype=np.uint8)
        cv2.rectangle(
            frame, (200, 100), (300, 200), (0, 0, 0), -1
        )  # Rectángulo como mano
        out.write(frame)

    out.release()
    return video_path, output_path


def test_capture_samples_from_video_crea_muestras_mock(dummy_video):
    """
    Verifica que `capture_samples_from_video` procese el video y cree muestras (con mocks).
    """
    video_path, output_path = dummy_video

    with patch(
        "ml.features.capture_samples_video.draw_keypoints", return_value=None
    ), patch(
        "ml.features.capture_samples_video.mediapipe_detection"
    ) as mock_detect, patch(
        "ml.features.capture_samples_video.Holistic"
    ) as MockHolistic:
        # Simula un modelo y sus resultados
        mock_model = MagicMock()
        MockHolistic.return_value.__enter__.return_value = mock_model

        mock_results_con_mano = MagicMock()
        mock_results_con_mano.left_hand_landmarks = True

        mock_results_sin_mano = MagicMock()
        mock_results_sin_mano.left_hand_landmarks = None

        # 30 frames: mitad con mano, mitad sin mano
        mock_detect.side_effect = [mock_results_con_mano] * 15 + [
            mock_results_sin_mano
        ] * 15

        with patch(
            "ml.features.capture_samples_video.there_hand",
            side_effect=[True] * 15 + [False] * 15,
        ):
            capture_samples_from_video(
                video_path=video_path,
                path=output_path,
                debug=False,
                margin_frames=1,
                min_frames=5,
                delay_frames=2,
            )

    muestras = [
        d
        for d in os.listdir(output_path)
        if os.path.isdir(os.path.join(output_path, d))
    ]

    assert len(muestras) >= 1, "No se creó ninguna carpeta de muestra"
    for carpeta in muestras:
        imgs = os.listdir(os.path.join(output_path, carpeta))
        assert len(imgs) > 0, f"La muestra {carpeta} está vacía"
