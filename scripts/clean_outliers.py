"""
Script para eliminar outliers de la base de datos.
Ejecutar con: python clean_outliers.py
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
from app.database.connection import get_connection


def delete_sample(sample_id: int):
    """Elimina una muestra y sus keypoints de la BD."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM keypoints WHERE sample_id = %s;", (sample_id,))
    cur.execute("DELETE FROM samples WHERE sample_id = %s;", (sample_id,))
    conn.commit()
    cur.close()
    conn.close()


def remove_outliers(threshold_std: float = 2.0, dry_run: bool = True):
    """
    Detecta y elimina outliers de la BD.

    Args:
        threshold_std: umbral en desviaciones estándar.
        dry_run: si True solo muestra qué eliminaría, sin borrar nada.
    """
    word_ids = fetch_word_ids_with_keypoints()

    print(f"\n{'='*60}")
    mode = "DRY RUN (simulación)" if dry_run else "ELIMINANDO OUTLIERS"
    print(f"Limpieza de outliers — {mode}")
    print(f"Umbral: {threshold_std} desviaciones estándar")
    print(f"{'='*60}\n")

    total_eliminados = 0

    for wid in word_ids:
        row = get_word_by_id(bytes(wid))
        word = row[1] if row else "?"

        raw = fetch_keypoints_for_words([bytes(wid)])

        grouped = {}
        for _, sample_id, frame, kp_json in raw:
            kp = (
                np.array(json.loads(kp_json))
                if isinstance(kp_json, str)
                else np.array(kp_json)
            )
            grouped.setdefault(sample_id, []).append((frame, kp))

        if len(grouped) < 3:
            continue

        max_len = max(len(frames) for frames in grouped.values())
        sample_vectors = {}
        for sample_id, frames in grouped.items():
            ordered = [kp for _, kp in sorted(frames, key=lambda x: x[0])]
            while len(ordered) < max_len:
                ordered.append(ordered[-1])
            sample_vectors[sample_id] = np.concatenate(ordered[:max_len])

        matrix = np.stack(list(sample_vectors.values()))
        centroid = np.mean(matrix, axis=0)
        distances = {
            sid: np.linalg.norm(vec - centroid) for sid, vec in sample_vectors.items()
        }

        dist_values = np.array(list(distances.values()))
        mean_dist = np.mean(dist_values)
        std_dist = np.std(dist_values)
        threshold = mean_dist + threshold_std * std_dist

        outliers = {sid: d for sid, d in distances.items() if d > threshold}

        if outliers:
            print(f"📝 '{word}': eliminando {len(outliers)} outliers")
            for sid, dist in sorted(outliers.items(), key=lambda x: x[1], reverse=True):
                print(
                    f"   {'🗑️  Eliminando' if not dry_run else '⚠️  Eliminaría'} sample_id={sid} (distancia={dist:.2f})"
                )
                if not dry_run:
                    delete_sample(sid)
                    total_eliminados += 1

    print(f"\n{'='*60}")
    if dry_run:
        print(
            f"DRY RUN completado. Para eliminar de verdad, ejecutar con dry_run=False"
        )
    else:
        print(f"✅ {total_eliminados} muestras eliminadas.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Primero corrés con dry_run=True para ver qué va a eliminar
    # Cuando estés seguro, cambiás a dry_run=False
    remove_outliers(threshold_std=2.0, dry_run=False)
