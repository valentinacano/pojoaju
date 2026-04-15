"""
Fixtures globales de pytest.
"""

import pytest
import psycopg2
from app.config import DB_CONFIG
from app.database.schema import create_all_tables


@pytest.fixture(scope="function")
def clean_db():
    """
    Limpia y recrea las tablas antes de cada test.

    Garantiza un entorno aislado sin datos residuales.
    """
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("SET session_replication_role = replica;")
    for table in ["keypoints", "samples", "words", "categories"]:
        cur.execute(f"TRUNCATE TABLE {table} RESTART IDENTITY CASCADE;")
    cur.execute("SET session_replication_role = DEFAULT;")
    conn.commit()
    cur.close()
    conn.close()

    create_all_tables()