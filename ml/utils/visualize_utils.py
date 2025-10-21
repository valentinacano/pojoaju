"""
Visualización de keypoints 2D como stickman y guardado como imagen PNG.

Este módulo permite dibujar un conjunto de keypoints bidimensionales conectados
por líneas (stickman) a partir de una secuencia normalizada. Es utilizado para
visualizar la representación promedio de una seña o palabra.

La función `draw_keypoints` genera una imagen `.png` que puede ser usada en la web
u otras interfaces para mostrar al usuario la forma general del gesto.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import hashlib

# Conexiones entre puntos del cuerpo (pueden adaptarse según el modelo usado)
BODY_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
]


def draw_keypoints(keypoints_2d, save_path="output.png", title="Seña"):
    """
    Dibuja los keypoints como un stickman y guarda la imagen como PNG.

    Recibe un array plano o lista de coordenadas `(x, y)`, lo convierte en
    un conjunto de puntos conectados por líneas, y guarda la visualización
    como una imagen PNG en la ruta especificada.

    Args:
        keypoints_2d (list | np.ndarray): Lista o array plano de coordenadas (x1, y1, x2, y2, ...).
        save_path (str): Ruta donde se guardará la imagen generada. Default: "output.png".
        title (str): Título opcional para mostrar en la imagen. Default: "Seña".

    Returns:
        None: La función guarda la imagen en disco, no retorna ningún valor.
    """
    keypoints = np.array(keypoints_2d).reshape(-1, 2)

    plt.figure(figsize=(4, 6))
    ax = plt.gca()
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)  # Invertimos eje Y para formato natural
    ax.axis("off")

    for x, y in keypoints:
        plt.scatter(x, y, c="blue", s=20)

    for i, j in BODY_CONNECTIONS:
        if i < len(keypoints) and j < len(keypoints):
            xi, yi = keypoints[i]
            xj, yj = keypoints[j]
            plt.plot([xi, xj], [yi, yj], c="black", linewidth=2)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
