from app.database.connection import get_connection
import json


def save_keypoints_to_db(word_id, sample_id, keypoints_sequence):
    """
    Guarda una secuencia de keypoints en la base de datos PostgreSQL.

    Cada vector de keypoints se inserta como una fila individual en la tabla `keypoints`,
    asociando el frame a un `word_id` y `sample_id`. Los vectores se almacenan en formato JSON.

    Args:
        word_id (int): Identificador de la palabra asociada a la muestra.
        sample_id (int): Identificador de la muestra.
        keypoints_sequence (list[np.ndarray]): Lista de vectores de keypoints por frame.

    Returns:
        None: Esta función no retorna nada. Inserta los datos directamente en la base.

    Raises:
        Exception: Propaga cualquier error que ocurra durante la conexión o inserción.
    """
    try:
        print(f"🟡 Recibidos {len(keypoints_sequence)} frames para sample {sample_id}")

        if len(keypoints_sequence) == 0:
            print("⚠️ No hay keypoints para insertar. Abortando.")
            return

        conn = get_connection()
        cur = conn.cursor()

        # Inserta cada frame individualmente
        for frame_index, keypoints_data in enumerate(keypoints_sequence, start=1):
            print(
                f"🟢 Insertando frame {frame_index} (sample {sample_id}, word_id {word_id})"
            )
            cur.execute(
                """
                INSERT INTO keypoints (word_id, sample, frame, keypoints)
                VALUES (%s, %s, %s, %s)
                RETURNING id;
            """,
                (word_id, sample_id, frame_index, json.dumps(keypoints_data.tolist())),
            )

        conn.commit()

        # Intenta recuperar el último ID insertado (si aplica)
        try:
            inserted_id = cur.fetchone()[0]
            print(f"🆔 ID insertado: {inserted_id}")
        except Exception:
            pass  # En caso de múltiples inserts, fetchone puede no aplicar

        cur.close()
        conn.close()

        print(f"✅ Insertados {len(keypoints_sequence)} frames en la base.")

    except Exception as e:
        print("❌ Error al insertar en base de datos:", e)
        raise
