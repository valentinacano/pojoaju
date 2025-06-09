from app.database.connection import get_connection
from ml.utils.common_utils import clean_word
import json
import hashlib


def _execute_query(
    query: str, params: tuple = None, fetch_one: bool = False, fetch_all: bool = False
):
    """
    Ejecuta una consulta SQL utilizando la conexión de base de datos.

    Permite ejecutar tanto consultas de modificación como de lectura, con manejo de errores.

    Args:
        query (str): Consulta SQL a ejecutar.
        params (tuple): Parámetros opcionales para la consulta SQL.
        fetch_one (bool): Si es True, retorna una sola fila.
        fetch_all (bool): Si es True, retorna todas las filas.

    Returns:
        any | None: Resultado(s) de la consulta si es SELECT; None en otros casos.
    """
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(query, params)

        result = None
        if fetch_one:
            result = cur.fetchone()
        elif fetch_all:
            result = cur.fetchall()

        conn.commit()
        cur.close()
        conn.close()

        return result
    except Exception as e:
        print("❌ Error al ejecutar la consulta:", e)
        raise


def insert_sample(word_id):
    """
    Inserta un nuevo registro en la tabla `samples` para una palabra dada.

    Args:
        word_id (bytes): Identificador único de la palabra (hash en formato BYTEA).

    Returns:
        int: ID del nuevo sample insertado.
    """
    query = """
        INSERT INTO samples (word_id)
        VALUES (%s)
        RETURNING sample_id;
    """
    return _execute_query(query, (word_id,), fetch_one=True)[0]


def insert_keypoints(word_id, sample_id, keypoints_sequence):
    """
    Guarda una secuencia de keypoints en la base de datos PostgreSQL.

    Inserta cada frame de una secuencia de keypoints como un registro individual
    en la tabla `keypoints`, asociando el frame a un `word_id` y `sample_id`.

    Args:
        word_id (bytes): Identificador único de la palabra asociada a la muestra.
        sample_id (int): Identificador único del sample insertado.
        keypoints_sequence (list[np.ndarray]): Lista de vectores de keypoints por frame.

    Returns:
        None: Esta función no retorna ningún valor. Inserta los datos directamente en la base.
    """
    if keypoints_sequence is None or len(keypoints_sequence) == 0:
        print("⚠️ No hay keypoints para insertar. Abortando.")
        return

    print(f"🟡 Recibidos {len(keypoints_sequence)} frames para sample {sample_id}")

    for frame_index, keypoints_data in enumerate(keypoints_sequence, start=1):
        print(
            f"🟢 Insertando frame {frame_index} (sample {sample_id}, word_id {word_id})"
        )
        query = """
            INSERT INTO keypoints (word_id, sample_id, frame, keypoints)
            VALUES (%s, %s, %s, %s);
        """
        params = (word_id, sample_id, frame_index, json.dumps(keypoints_data.tolist()))
        _execute_query(query, params)

    print(f"✅ Insertados {len(keypoints_sequence)} frames en la base.")


def insert_words(words):
    """
    Inserta palabras en la tabla `words` agrupadas por categoría.

    Limpia y normaliza las palabras antes de insertar, evitando duplicados.
    Crea las categorías si no existen, y usa un hash para identificar cada palabra de forma única.

    Args:
        words (dict): Diccionario donde las claves son categorías y los valores son listas de palabras.

    Returns:
        None: Esta función no retorna ningún valor.
    """
    for category, word_list in words.items():
        category_clean = clean_word(category)
        _execute_query(
            "INSERT INTO categories (category) VALUES (%s) ON CONFLICT (category) DO NOTHING;",
            (category_clean,),
        )

        result = _execute_query(
            "SELECT category_id FROM categories WHERE category = %s;",
            (category_clean,),
            fetch_one=True,
        )
        category_id = result[0]

        for word in word_list:
            word_clean = clean_word(word)
            word_id = hashlib.sha256(word_clean.encode("utf-8")).digest()
            _execute_query(
                "INSERT INTO words (word_id, category_id, word) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING;",
                (word_id, category_id, word_clean),
            )

    print("✅ Palabras insertadas en la tabla 'words'.")


def insert_categories(categories):
    """
    Inserta categorías en la tabla `categories` si no existen.

    Limpia y normaliza las categorías antes de insertar, evitando duplicados.

    Args:
        categories (list): Lista de categorías a insertar.

    Returns:
        None: Esta función no retorna ningún valor.
    """
    for category in categories:
        category_clean = clean_word(category)
        _execute_query(
            "INSERT INTO categories (category) VALUES (%s) ON CONFLICT (category) DO NOTHING;",
            (category_clean,),
        )
    print("✅ Categorías insertadas correctamente.")


def fetch_all_words():
    """
    Obtiene todas las palabras de la base de datos junto con sus categorías.

    Returns:
        list[tuple] | []: Lista de tuplas (word_id, word, category).
    """
    query = """
        SELECT w.word_id, w.word, c.category
        FROM words w
        JOIN categories c ON w.category_id = c.category_id
        ORDER BY c.category, w.word;
    """
    return _execute_query(query, fetch_all=True) or []


def search_word(word):
    """
    Busca una palabra en la base de datos y retorna su información asociada.

    Args:
        word (str): Palabra a buscar.

    Returns:
        tuple | None: Tupla con word_id, palabra y categoría si se encuentra; None en caso contrario.
    """
    word_clean = clean_word(word)
    word_id = hashlib.sha256(word_clean.encode("utf-8")).digest()
    query = """
        SELECT w.word_id, w.word, c.category
        FROM words w
        JOIN categories c ON w.category_id = c.category_id
        WHERE w.word_id = %s;
    """
    return _execute_query(query, (word_id,), fetch_one=True)


def fetch_all_categories():
    """
    Obtiene todas las categorías existentes en la base de datos.

    Returns:
        list[str]: Lista de nombres de categorías ordenadas alfabéticamente.
    """
    query = "SELECT category FROM categories ORDER BY category;"
    result = _execute_query(query, fetch_all=True)
    return [row[0] for row in result] if result else []
