from app.database.connection import get_connection


def test_database_connection():
    """
    Verifica que se pueda establecer una conexión a la base de datos.

    Este test usa los parámetros definidos en `DB_CONFIG` y verifica que la conexión
    se haya establecido correctamente, asegurándose de que esté abierta.

    Args:
        None

    Returns:
        None: Lanza una excepción si la conexión falla o está cerrada.
    """
    conn = get_connection()
    assert conn is not None, "No se pudo establecer conexión"
    assert conn.closed == 0, "La conexión está cerrada"
    conn.close()
