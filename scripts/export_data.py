"""
Exporta samples y keypoints a un archivo JSON para compartir entre compañeros.

Ejecutar con: python scripts/export_data.py
El archivo generado se llama: pojoaju_export.json
"""

from dotenv import load_dotenv

load_dotenv()

import json
from app.database.connection import get_connection
from app.database.queries import fetch_word_ids_with_keypoints, get_word_by_id


def export_data(output_file: str = "pojoaju_export.json"):
    """
    Exporta todos los samples y keypoints de la BD a un archivo JSON.
    Incluye el nombre de la palabra para que el importador pueda
    hacer el match correcto en la BD del destinatario.
    """
    conn = get_connection()
    cur = conn.cursor()

    print("📦 Exportando datos...")

    # Obtener todos los samples con sus keypoints
    cur.execute(
        """
        SELECT s.sample_id, w.word, k.frame, k.keypoints
        FROM samples s
        JOIN words w ON s.word_id = w.word_id
        JOIN keypoints k ON k.sample_id = s.sample_id
        ORDER BY w.word, s.sample_id, k.frame;
    """
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    if not rows:
        print("⚠️ No hay datos para exportar.")
        return

    # Agrupar por (word, sample_id)
    grouped = {}
    for sample_id, word, frame, keypoints in rows:
        key = (word, sample_id)
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(
            {
                "frame": frame,
                "keypoints": (
                    keypoints if isinstance(keypoints, list) else json.loads(keypoints)
                ),
            }
        )

    # Construir estructura de exportación
    export = []
    for (word, sample_id), frames in grouped.items():
        export.append(
            {"word": word, "frames": sorted(frames, key=lambda x: x["frame"])}
        )

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(export, f, ensure_ascii=False)

    words_count = len(set(e["word"] for e in export))
    print(f"✅ Exportados {len(export)} samples de {words_count} palabras.")
    print(f"📁 Archivo generado: {output_file}")
    print(f"📧 Mandá este archivo a tu compañero.")


if __name__ == "__main__":
    export_data()
