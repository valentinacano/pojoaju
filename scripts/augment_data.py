"""
Generación de muestras sintéticas por data augmentation.

Toma las muestras reales de la BD, aplica variaciones pequeñas
y las guarda como nuevas muestras. Útil para aumentar el dataset
sin necesidad de capturar más videos.

Técnicas aplicadas:
- Ruido gaussiano (simula pequeñas variaciones de posición)
- Escala leve (simula distancia diferente a la cámara)
- Desplazamiento (simula posición diferente en el encuadre)

Ejecutar con: python scripts/augment_data.py
"""

from dotenv import load_dotenv
load_dotenv()

import json
import numpy as np
from app.database.queries import (
    fetch_word_ids_with_keypoints,
    fetch_keypoints_for_words,
    get_word_by_id,
    word_to_id,
    insert_sample,
    insert_keypoints,
)
from app.database.connection import get_connection
from ml.normalize import normalize_sequence
from app.config import MODEL_FRAMES


# ---------------------------------------------------------------------------
# Técnicas de augmentación
# ---------------------------------------------------------------------------

def augment_noise(sequence: np.ndarray, noise_std: float = 0.005) -> np.ndarray:
    """
    Agrega ruido gaussiano pequeño a la secuencia.
    Simula variaciones naturales en la posición de los landmarks.
    """
    noise = np.random.normal(0, noise_std, sequence.shape)
    return np.clip(sequence + noise, 0, 1)


def augment_scale(sequence: np.ndarray, scale_range: tuple = (0.95, 1.05)) -> np.ndarray:
    """
    Aplica un factor de escala aleatorio alrededor del centroide.
    Simula diferente distancia a la cámara.
    """
    scale = np.random.uniform(*scale_range)
    center = np.mean(sequence, axis=0)
    return np.clip(center + scale * (sequence - center), 0, 1)


def augment_shift(sequence: np.ndarray, shift_range: float = 0.02) -> np.ndarray:
    """
    Desplaza toda la secuencia levemente en x e y.
    Simula diferente posición en el encuadre.
    """
    shift_x = np.random.uniform(-shift_range, shift_range)
    shift_y = np.random.uniform(-shift_range, shift_range)
    shifted = sequence.copy()
    # Desplazar solo las coordenadas x e y (cada 3-4 valores según el landmark)
    shifted += np.random.uniform(-shift_range, shift_range, sequence.shape) * 0.5
    return np.clip(shifted, 0, 1)


def augment_sequence(sequence: np.ndarray) -> np.ndarray:
    """
    Aplica una combinación aleatoria de augmentaciones a una secuencia.

    Args:
        sequence: array de shape (MODEL_FRAMES, LENGTH_KEYPOINTS)

    Returns:
        Secuencia augmentada del mismo shape.
    """
    aug = sequence.copy()

    # Siempre aplica ruido
    aug = augment_noise(aug, noise_std=np.random.uniform(0.003, 0.008))

    # 70% de probabilidad de escala
    if np.random.random() < 0.7:
        aug = augment_scale(aug, scale_range=(0.93, 1.07))

    # 50% de probabilidad de shift
    if np.random.random() < 0.5:
        aug = augment_shift(aug, shift_range=0.015)

    return aug


# ---------------------------------------------------------------------------
# Carga de muestras reales
# ---------------------------------------------------------------------------

def load_real_samples(word_id: bytes) -> list[np.ndarray]:
    """
    Carga todas las muestras reales de una palabra desde la BD.

    Returns:
        Lista de arrays de shape (MODEL_FRAMES, LENGTH_KEYPOINTS)
    """
    raw = fetch_keypoints_for_words([word_id])

    grouped = {}
    for _, sample_id, frame, kp_json in raw:
        kp = np.array(json.loads(kp_json)) if isinstance(kp_json, str) else np.array(kp_json)
        grouped.setdefault(sample_id, []).append((frame, kp))

    sequences = []
    for sample_id in sorted(grouped.keys()):
        frames = grouped[sample_id]
        ordered = [kp for _, kp in sorted(frames, key=lambda x: x[0])]
        normalized = normalize_sequence(ordered, MODEL_FRAMES)
        sequences.append(normalized)

    return sequences


# ---------------------------------------------------------------------------
# Generación y guardado
# ---------------------------------------------------------------------------

def generate_synthetic_samples(
    word: str,
    target: int = 100,
    dry_run: bool = True
):
    """
    Genera muestras sintéticas para una palabra hasta alcanzar el target.

    Args:
        word: nombre de la palabra.
        target: cantidad total de muestras deseada (reales + sintéticas).
        dry_run: si True solo muestra qué generaría, sin insertar nada.
    """
    wid = word_to_id(word)
    real_samples = load_real_samples(wid)
    current = len(real_samples)
    needed = max(0, target - current)

    if needed == 0:
        print(f"✅ '{word}': ya tiene {current} muestras, no necesita augmentación.")
        return

    print(f"📝 '{word}': {current} reales → generando {needed} sintéticas para llegar a {target}")

    if dry_run:
        print(f"   [DRY RUN] Se insertarían {needed} muestras sintéticas.")
        return

    inserted = 0
    for i in range(needed):
        # Elegir una muestra real aleatoria como base
        base = real_samples[np.random.randint(len(real_samples))]
        synthetic = augment_sequence(base)

        sample_id = insert_sample(wid)
        insert_keypoints(wid, sample_id, list(synthetic))
        inserted += 1

        if (i + 1) % 10 == 0:
            print(f"   {i + 1}/{needed} muestras generadas...")

    print(f"   ✅ {inserted} muestras sintéticas insertadas para '{word}'.")


def augment_all(target: int = 100, dry_run: bool = True):
    """
    Genera muestras sintéticas para todas las palabras con keypoints.

    Args:
        target: cantidad total de muestras deseada por palabra.
        dry_run: si True solo muestra qué generaría.
    """
    word_ids = fetch_word_ids_with_keypoints()

    print(f"\n{'='*60}")
    mode = "DRY RUN (simulación)" if dry_run else "GENERANDO MUESTRAS SINTÉTICAS"
    print(f"Data Augmentation — {mode}")
    print(f"Target: {target} muestras por palabra")
    print(f"{'='*60}\n")

    for wid in word_ids:
        row = get_word_by_id(bytes(wid))
        word = row[1] if row else "?"
        generate_synthetic_samples(word, target=target, dry_run=dry_run)

    print(f"\n{'='*60}")
    if dry_run:
        print("DRY RUN completado. Para generar de verdad, cambiá dry_run=False.")
    else:
        print("✅ Augmentación completada.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Primero corré con dry_run=True para ver qué va a generar
    # Cuando estés conforme, cambiá a dry_run=False
    augment_all(target=120, dry_run=False)