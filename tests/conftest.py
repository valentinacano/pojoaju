"""
Fixture de pytest para limpiar la base de datos antes de ejecutar tests.

Este módulo define una función `clean_test_database` que se encarga de 
resetear las tablas principales (`categories`, `words`, `samples`, `keypoints`)
dejando la base en un estado limpio para pruebas controladas.
"""


import pytest
import psycopg2
from app.config import DB_CONFIG


@pytest.fixture(scope="function")
def clean_test_database():
    """
    Limpia las tablas principales de la base de datos de prueba antes de cada test.

    Esta fixture se conecta a la base de datos definida en `DB_CONFIG`, desactiva
    temporalmente las restricciones de claves foráneas y trunca las tablas
    `keypoints`, `samples`, `words` y `categories` para garantizar un entorno
    limpio y consistente en cada ejecución de prueba.

    Returns:
        None: Esta función no retorna ningún valor. Modifica directamente
        el contenido de la base de datos.
    """
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # Desactiva claves foráneas para truncar sin conflictos
    cur.execute("SET session_replication_role = replica;")

    # Truncado en orden para evitar errores de dependencia
    tables = ["keypoints", "samples", "words", "categories"]
    for table in tables:
        cur.execute(f"TRUNCATE TABLE {table} RESTART IDENTITY CASCADE;")

    # Reactiva las restricciones
    cur.execute("SET session_replication_role = DEFAULT;")

    conn.commit()
    cur.close()
    conn.close()
