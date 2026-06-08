"""
Todas las queries a PostgreSQL en un único módulo.

Reglas:
- Cada función abre y cierra su propia conexión.
- Nunca retorna objetos de conexión al exterior.
- Los word_id son siempre bytes (SHA256 digest).
"""

import json
import hashlib
import numpy as np

from app.database.connection import get_connection


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------


def _exec(query: str, params=None, fetch_one=False, fetch_all=False):
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


def word_to_id(word: str) -> bytes:
    """Convierte una palabra a su word_id (SHA256 digest)."""
    return hashlib.sha256(word.strip().lower().encode()).digest()


# ---------------------------------------------------------------------------
# Categories
# ---------------------------------------------------------------------------


def insert_category(category: str):
    _exec(
        "INSERT INTO categories (category) VALUES (%s) ON CONFLICT (category) DO NOTHING;",
        (category.strip().lower(),),
    )


def get_category_id(category: str) -> int | None:
    row = _exec(
        "SELECT category_id FROM categories WHERE category = %s;",
        (category.strip().lower(),),
        fetch_one=True,
    )
    return row[0] if row else None


def fetch_all_categories() -> list[str]:
    rows = _exec("SELECT category FROM categories ORDER BY category;", fetch_all=True)
    return [r[0] for r in rows] if rows else []


# ---------------------------------------------------------------------------
# Words
# ---------------------------------------------------------------------------


def insert_word(word: str, category: str):
    """
    Inserta una palabra con su categoría. Crea la categoría si no existe.
    """
    cat = category.strip().lower()
    _exec(
        "INSERT INTO categories (category) VALUES (%s) ON CONFLICT (category) DO NOTHING;",
        (cat,),
    )
    cat_id = get_category_id(cat)
    wid = word_to_id(word)
    _exec(
        "INSERT INTO words (word_id, category_id, word) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING;",
        (wid, cat_id, word.strip().lower()),
    )


def insert_words_bulk(words: dict):
    """
    Inserta múltiples palabras desde un dict {categoria: [palabras]}.
    """
    for category, word_list in words.items():
        for word in word_list:
            insert_word(word, category)
    print(f"✅ {sum(len(v) for v in words.values())} palabras insertadas.")


def fetch_all_words() -> list[tuple]:
    """Retorna lista de (word_id, word, category)."""
    return (
        _exec(
            """
        SELECT w.word_id, w.word, c.category
        FROM words w
        JOIN categories c ON w.category_id = c.category_id
        ORDER BY c.category, w.word;
    """,
            fetch_all=True,
        )
        or []
    )


def get_word_by_id(word_id: bytes) -> tuple | None:
    """Retorna (word_id, word, category) o None."""
    return _exec(
        """
        SELECT w.word_id, w.word, c.category
        FROM words w
        JOIN categories c ON w.category_id = c.category_id
        WHERE w.word_id = %s;
    """,
        (word_id,),
        fetch_one=True,
    )


def get_word_by_name(word: str) -> tuple | None:
    """Retorna (word_id, word, category) o None."""
    return get_word_by_id(word_to_id(word))


# ---------------------------------------------------------------------------
# Samples
# ---------------------------------------------------------------------------


def insert_sample(word_id: bytes) -> int:
    """Inserta una muestra y retorna su sample_id."""
    row = _exec(
        "INSERT INTO samples (word_id) VALUES (%s) RETURNING sample_id;",
        (word_id,),
        fetch_one=True,
    )
    return row[0]


# ---------------------------------------------------------------------------
# Keypoints
# ---------------------------------------------------------------------------


def insert_keypoints(word_id: bytes, sample_id: int, sequence: list):
    """
    Guarda una secuencia de keypoints frame por frame.

    Args:
        word_id: ID de la palabra.
        sample_id: ID de la muestra.
        sequence: lista de np.ndarray, uno por frame.
    """
    if sequence is None or len(sequence) == 0:
        print("⚠️ Secuencia vacía, no se insertan keypoints.")
        return

    for frame_idx, keypoints in enumerate(sequence, start=1):
        kp = keypoints.tolist() if hasattr(keypoints, "tolist") else keypoints
        _exec(
            "INSERT INTO keypoints (word_id, sample_id, frame, keypoints) VALUES (%s, %s, %s, %s);",
            (word_id, sample_id, frame_idx, json.dumps(kp)),
        )
    print(f"✅ {len(sequence)} frames insertados (sample {sample_id})")


def fetch_keypoints_for_words(word_ids: list[bytes]) -> list[tuple]:
    """
    Retorna todos los registros de keypoints para una lista de word_ids.

    Returns:
        list de (word_id, sample_id, frame, keypoints_json_str)
    """
    if not word_ids:
        return []
    placeholders = ",".join(["%s"] * len(word_ids))
    return (
        _exec(
            f"SELECT word_id, sample_id, frame, keypoints FROM keypoints WHERE word_id IN ({placeholders}) ORDER BY word_id, sample_id, frame;",
            tuple(word_ids),
            fetch_all=True,
        )
        or []
    )


def fetch_word_ids_with_keypoints() -> list[bytes]:
    """Retorna los word_ids que tienen al menos un keypoint registrado."""
    rows = _exec(
        "SELECT DISTINCT word_id FROM keypoints ORDER BY word_id;", fetch_all=True
    )
    return [r[0] for r in rows] if rows else []


def count_samples_per_word(word_ids: list[bytes]) -> dict:
    """Retorna {word_id: cantidad_de_samples_distintos} con una sola query eficiente."""
    if not word_ids:
        return {}

    placeholders = ",".join(["%s"] * len(word_ids))
    rows = (
        _exec(
            f"""
        SELECT word_id, COUNT(*)
        FROM samples
        WHERE word_id IN ({placeholders})
        GROUP BY word_id
        """,
            params=tuple(word_ids),
            fetch_all=True,
        )
        or []
    )

    return {word_id: count for word_id, count in rows}
