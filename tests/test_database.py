from app.database.connection import get_connection
from app.config import DB_CONFIG


def test_database_connection():
    """
    Verifica que se establezca una conexión activa a la base de datos
    y que corresponda al nombre esperado en `DB_CONFIG`.

    Args:
        None

    Returns:
        None
    """
    conn = get_connection()
    dbname_expected = "pojoaju_test"
    dbname_actual = conn.get_dsn_parameters().get("dbname")

    assert conn is not None, "La conexión retornada es None"
    assert conn.closed == 0, "La conexión está cerrada"
    assert (
        dbname_actual == dbname_expected
    ), f"Se esperaba conexión a '{dbname_expected}', pero se conectó a '{dbname_actual}'"

    conn.close()
