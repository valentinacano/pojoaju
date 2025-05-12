from app.database.connection import get_connection
from ml.utils.common_utils import clean_word
import json
import hashlib


def insert_keypoints(word_id, sample_id, keypoints_sequence):
    """
    Guarda una secuencia de keypoints en la base de datos PostgreSQL.

    Inserta cada frame de una secuencia de keypoints como un registro individual
    en la tabla `keypoints`, asociando el frame a un `word_id` y `sample_id`.

    Args:
        word_id (int): Identificador único de la palabra asociada a la muestra.
        sample_id (int): Identificador único de la muestra.
        keypoints_sequence (list[np.ndarray]): Lista de vectores de keypoints por frame.

    Returns:
        None: Esta función no retorna ningún valor. Inserta los datos directamente en la base.

    Raises:
        Exception: Lanza cualquier error que ocurra durante la conexión o inserción.
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
                RETURNING keypoints_id;
            """,
                (word_id, sample_id, frame_index, json.dumps(keypoints_data.tolist())),
            )

        conn.commit()
        cur.close()
        conn.close()

        print(f"✅ Insertados {len(keypoints_sequence)} frames en la base.")

    except Exception as e:
        print("❌ Error al insertar en base de datos:", e)
        raise


def insert_words(words):
    """
    Inserta palabras en la tabla `words` agrupadas por categoría.

    Limpia y normaliza las palabras antes de insertar, evitando duplicados.
    Crea las categorías si no existen, y usa un hash para identificar cada palabra de forma única.

    Args:
        words (dict): Diccionario donde las claves son categorías y los valores son listas de palabras.

    Returns:
        None: Esta función no retorna ningún valor.

    Raises:
        Exception: Lanza cualquier error que ocurra durante la inserción de palabras.
    """
    try:
        conn = get_connection()
        cur = conn.cursor()

        # Limpiar y normalizar palabras
        clean_words = {}
        for category, word_list in words.items():
            cleaned_category = clean_word(category)
            cleaned_words = set([clean_word(w) for w in word_list if w])
            if cleaned_category:
                clean_words[cleaned_category] = list(cleaned_words)

        for category, word_list in clean_words.items():
            # Inserta la categoría si no existe
            cur.execute(
                "INSERT INTO categories (category) VALUES (%s) ON CONFLICT (category) DO NOTHING;",
                (category,),
            )
            # Obtiene el ID de la categoría
            cur.execute(
                "SELECT category_id FROM categories WHERE category = %s;", (category,)
            )
            category_id = cur.fetchone()[0]

            # Inserta cada palabra con hash único
            for word in word_list:
                clean_word_value = clean_word(word)
                hash_value = hashlib.sha256(clean_word_value.encode("utf-8")).digest()
                query = "INSERT INTO words (word_id, category_id, word) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING;"
                cur.execute(query, (hash_value, category_id, clean_word_value))

        conn.commit()
        cur.close()
        conn.close()
        print("✅ Palabras insertadas en la tabla 'words'.")
    except Exception as e:
        print("❌ Error al insertar palabras:", e)
        raise


def insert_categories(categories):
    """
    Inserta categorías en la tabla `categories` si no existen.

    Limpia y normaliza las categorías antes de insertar, evitando duplicados.

    Args:
        categories (list): Lista de categorías a insertar.

    Returns:
        None: Esta función no retorna ningún valor.

    Raises:
        Exception: Lanza cualquier error que ocurra durante la inserción de categorías.
    """
    try:
        conn = get_connection()
        cur = conn.cursor()

        # Limpiar y normalizar categorías
        clean_categories = set([clean_word(cat) for cat in categories if cat.strip()])
        for category in clean_categories:
            cur.execute(
                "INSERT INTO categories (category) VALUES (%s) ON CONFLICT (category) DO NOTHING;",
                (category,),
            )

        conn.commit()
        cur.close()
        conn.close()
        print("✅ Categorías insertadas correctamente.")
    except Exception as e:
        print("❌ Error al insertar categorías:", e)
        raise
