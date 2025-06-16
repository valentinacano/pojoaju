import tempfile, pytest, cv2
import numpy as np
from unittest.mock import patch

from ml.features.pipelines import create_samples_from_camera, save_keypoints
from ml.utils.common_utils import create_folder
from app.database.database_utils import fetch_keypoints_by_words
from app.database.connection import get_connection


def test_database_connection():
    """
    Verifica que se establezca una conexión activa a la base de datos
    y que corresponda al nombre esperado en `DB_CONFIG`.

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
    if dbname_actual != "pojoaju_test":
        pytest.exit("❌ Conectado a una base que no es de pruebas. Deteniendo tests.")
    conn.close()


@pytest.fixture
def dummy_sample_folder(tmp_path):
    """
    Crea una carpeta temporal con una estructura de muestra para pruebas.

    Genera una carpeta con nombre de palabra que contiene subcarpetas `sample_XX`
    con imágenes dummy que simulan frames capturados.

    Args:
        tmp_path (Path): Carpeta temporal proporcionada por pytest.

    Returns:
        Path: Ruta al directorio base de muestras.
    """
    word_name = "hola"
    base_path = tmp_path / word_name
    create_folder(base_path)

    for i in range(2):
        sample_dir = base_path / f"sample_0{i+1}"
        sample_dir.mkdir()
        for j in range(10):
            img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            cv2.imwrite(str(sample_dir / f"frame_{j:02}.jpg"), img)

    return base_path


def test_save_keypoints_inserta_en_db(dummy_sample_folder):
    """
    Verifica que `save_keypoints` extrae y guarda keypoints en base de datos.

    Utiliza una carpeta dummy con imágenes, ejecuta el pipeline de normalización
    y extracción, y confirma que los keypoints se insertan correctamente.

    Args:
        dummy_sample_folder (Path): Fixture con estructura de muestras.

    Returns:
        None
    """
    from hashlib import sha256

    word_name = dummy_sample_folder.name
    root_path = str(dummy_sample_folder.parent)
    word_id = sha256(word_name.encode()).digest()

    # Insertar sample y keypoints
    save_keypoints(word_name, word_id, root_path)

    # Validar que se insertaron keypoints
    resultados = fetch_keypoints_by_words([word_id])
    assert len(resultados) > 0, "No se insertaron keypoints en la base de datos"


def test_create_samples_from_camera_debug_true():
    """
    Verifica que `create_samples_from_camera` ejecuta el generador completo en modo debug.

    Mockea la función `capture_samples_from_camera` para simular la captura
    y asegura que se haya llamado internamente cuando debug=True.

    Returns:
        None
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        word_name = "prueba"

        # Simula que el generador produce 3 imágenes
        fake_generator = (np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(3))

        with patch(
            "ml.features.pipelines.capture_samples_from_camera",
            return_value=fake_generator,
        ) as mock_func:
            result = create_samples_from_camera(word_name, tmpdir, debug_value=True)

        assert mock_func.called, "No se llamó a capture_samples_from_camera"
        assert result is None, "No debe retornar nada en modo debug=True"


def test_create_samples_from_camera_debug_false():
    """
    Verifica que `create_samples_from_camera` retorne el generador en modo Flask.

    Mockea la función `capture_samples_from_camera` y confirma que se retorna
    correctamente el generador.

    Returns:
        None
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        word_name = "stream"
        dummy_gen = (b"frame" for _ in range(2))

        with patch(
            "ml.features.pipelines.capture_samples_from_camera", return_value=dummy_gen
        ):
            result = create_samples_from_camera(word_name, tmpdir, debug_value=False)

        assert hasattr(result, "__iter__"), "No se retornó un generador en modo Flask"
        assert next(result) == b"frame", "El generador no produce el valor esperado"
