import matplotlib.pyplot as plt
import numpy as np
import os
import hashlib

# Conexiones (ajustar según el modelo)
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
    Dibuja los keypoints como un stickman en una imagen PNG.
    """
    keypoints = np.array(keypoints_2d).reshape(-1, 2)

    plt.figure(figsize=(4, 6))
    ax = plt.gca()
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
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
