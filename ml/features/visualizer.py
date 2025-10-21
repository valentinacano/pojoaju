"""
Visualización de keypoints promedios para una palabra en imagen PNG.

Este módulo toma un conjunto de keypoints 2D (normalmente el promedio de muestras)
y genera una imagen representativa en formato `.png`, ideal para mostrar
cómo se vería una seña estándar para una palabra.

- Utiliza `matplotlib` para graficar.
- Usa `draw_keypoints` para dibujar la figura.
- Genera un hash SHA256 de la palabra para evitar nombres duplicados.
- Guarda la imagen en la carpeta `static/senas/`.

Función principal:
- `visualize_keypoints`: recibe los keypoints y guarda el gráfico como imagen PNG.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import hashlib

from ml.utils.visualize_utils import draw_keypoints


# Conexiones básicas entre puntos (puede variar según tu modelo: Mediapipe, etc.)
def visualize_keypoints(word, keypoints_2d, output_dir="static/senas"):
    """
    Genera y guarda la imagen PNG de los keypoints para una palabra.

    Args:
        word (str): Palabra asociada.
        keypoints_2d (list | np.ndarray): Keypoints promedio (x1, y1, x2, y2, ...)
        output_dir (str): Carpeta donde guardar la imagen.

    Returns:
        str: Ruta relativa al archivo generado.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Usamos el hash SHA256 como nombre de archivo (como tu sistema)
    word_hash = hashlib.sha256(word.strip().lower().encode("utf-8")).hexdigest()
    save_path = os.path.join(output_dir, f"{word_hash}.png")

    if not os.path.exists(save_path):
        draw_keypoints(keypoints_2d, save_path=save_path, title=f"Seña: {word}")
        print(f"✅ Imagen creada: {save_path}")
    else:
        print(f"ℹ️ Imagen ya existente: {save_path}")

    return f"/{output_dir}/{word_hash}.png"
