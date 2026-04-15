"""
Script temporal para analizar outliers en las muestras.
Ejecutar con: python check_outliers.py
"""
from dotenv import load_dotenv
load_dotenv()

import json
import numpy as np
from app.database.queries import (
    fetch_word_ids_with_keypoints,
    fetch_keypoints_for_words,
    get_word_by_id,
)
from dotenv import load_dotenv
load_dotenv()


def analyze_outliers(threshold_std: float = 2.0):
    """
    Analiza la distribución de distancias al centroide por palabra.
    Muestra las muestras candidatas a ser outliers.

    Args:
        threshold_std: muestras a más de N desviaciones estándar se consideran outliers.
    """
    word_ids = fetch_word_ids_with_keypoints()

    print(f"\n{'='*60}")
    print(f"Análisis de outliers (umbral: {threshold_std} desviaciones estándar)")
    print(f"{'='*60}\n")

    total_outliers = 0

    for wid in word_ids:
        row = get_word_by_id(bytes(wid))
        word = row[1] if row else "?"

        raw = fetch_keypoints_for_words([bytes(wid)])

        # Agrupar por sample_id
        grouped = {}
        for _, sample_id, frame, kp_json in raw:
            kp = np.array(json.loads(kp_json)) if isinstance(kp_json, str) else np.array(kp_json)
            grouped.setdefault(sample_id, []).append((frame, kp))

        # Construir vector por muestra (concatenar frames ordenados)
        sample_vectors = {}
        max_len = max(len(frames) for frames in grouped.values())
        for sample_id, frames in grouped.items():
            ordered = [kp for _, kp in sorted(frames, key=lambda x: x[0])]
            while len(ordered) < max_len:
                ordered.append(ordered[-1])
            sample_vectors[sample_id] = np.concatenate(ordered[:max_len])

        if len(sample_vectors) < 3:
            print(f"⚠️  '{word}': muy pocas muestras para analizar ({len(sample_vectors)})")
            continue

        # Calcular centroide y distancias
        matrix = np.stack(list(sample_vectors.values()))
        centroid = np.mean(matrix, axis=0)
        distances = {sid: np.linalg.norm(vec - centroid)
                     for sid, vec in sample_vectors.items()}

        dist_values = np.array(list(distances.values()))
        mean_dist = np.mean(dist_values)
        std_dist = np.std(dist_values)
        threshold = mean_dist + threshold_std * std_dist

        outliers = {sid: d for sid, d in distances.items() if d > threshold}
        total_outliers += len(outliers)

        print(f"📝 '{word}': {len(sample_vectors)} muestras")
        print(f"   Distancia promedio: {mean_dist:.2f} ± {std_dist:.2f}")
        print(f"   Umbral outlier:     {threshold:.2f}")
        print(f"   Outliers encontrados: {len(outliers)}")

        if outliers:
            for sid, dist in sorted(outliers.items(), key=lambda x: x[1], reverse=True):
                print(f"   ⚠️  sample_id={sid} → distancia={dist:.2f}")
        print()

    print(f"{'='*60}")
    print(f"Total outliers encontrados: {total_outliers}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    analyze_outliers(threshold_std=2.0)