"""
Tests de rutas y comportamiento de la aplicación Flask.

Este módulo verifica el correcto funcionamiento de las rutas de la interfaz web
incluidas en `app.views.flask_gui`. Se testean rutas de carga de plantillas, 
formularios, flujos de captura y respuestas esperadas del servidor.

Incluye también validaciones de errores comunes, redirecciones y comportamiento 
frente a rutas inexistentes.
"""

import pytest
from app.views.flask_gui import app


@pytest.fixture
def client():
    """
    Crea un cliente de prueba para la app Flask.

    Returns:
        flask.testing.FlaskClient: Cliente de prueba.
    """
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_index_route(client):
    """
    Verifica que la página principal se carga correctamente.

    Returns:
        None
    """
    response = client.get("/")
    assert response.status_code == 200
    assert b"<!DOCTYPE html" in response.data


def test_training_route(client):
    """
    Verifica que la ruta /training devuelve correctamente la plantilla.

    Returns:
        None
    """
    response = client.get("/training")
    assert response.status_code == 200
    assert b"<!DOCTYPE html" in response.data


def test_capture_form_route(client):
    """
    Verifica que la ruta /capture_form devuelve correctamente la plantilla.

    Returns:
        None
    """
    response = client.get("/capture_form")
    assert response.status_code == 200
    assert b"<!DOCTYPE html" in response.data


def test_dictionary_route(client):
    """
    Verifica que la ruta /training/dictionary devuelve una respuesta válida.

    Returns:
        None
    """
    response = client.get("/training/dictionary")
    assert response.status_code == 200
    assert b"<!DOCTYPE html" in response.data


def test_insert_word_form_get(client):
    """
    Verifica que la ruta GET /training/insert_word carga correctamente el formulario.

    Returns:
        None
    """
    response = client.get("/training/insert_word")
    assert response.status_code == 200
    assert b"<!DOCTYPE html" in response.data


def test_insert_word_form_post_missing_fields(client):
    """
    Verifica el comportamiento al enviar el formulario sin la palabra.

    Returns:
        None
    """
    response = client.post(
        "/training/insert_word",
        data={"word": "", "category_existing": "", "category_new": ""},
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert b"La palabra es obligatoria" in response.data


def test_insert_word_form_post_missing_category(client):
    """
    Verifica el comportamiento al no seleccionar ni ingresar una categoría.

    Returns:
        None
    """
    response = client.post(
        "/training/insert_word",
        data={"word": "hola", "category_existing": "", "category_new": ""},
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert b"Debe seleccionar o ingresar una categor" in response.data


def test_capture_route(client):
    """
    Verifica que la página de captura carga correctamente con parámetros válidos.

    Returns:
        None
    """
    response = client.get("/training/capture/testid/testword")
    assert response.status_code == 200
    assert b"testword" in response.data


def test_stop_capture_without_data(client):
    """
    Verifica que redirige correctamente si no se envían datos de word y word_id.

    Returns:
        None
    """
    response = client.post("/stop_capture", data={}, follow_redirects=True)
    assert response.status_code == 200
    assert b"<!DOCTYPE html" in response.data  # Redirige a /training


def test_stop_capture_with_data(client):
    """
    Verifica que redirige a /save_samples si se envían datos válidos.

    Returns:
        None
    """
    response = client.post(
        "/stop_capture",
        data={"word": "hola", "word_id": "abc123"},
        follow_redirects=False,
    )
    assert response.status_code == 302
    assert "/save_samples/hola/abc123" in response.headers["Location"]


def test_video_feed_route(client):
    """
    Verifica que la ruta de video_feed devuelve una respuesta válida con mimetype esperado.

    Returns:
        None
    """
    response = client.get("/video_feed/test")
    assert response.status_code == 200
    assert response.mimetype == "multipart/x-mixed-replace"


def test_save_samples_route(client, monkeypatch):
    """
    Verifica que la ruta /save_samples ejecuta correctamente el flujo y carga la plantilla.

    Returns:
        None
    """

    def fake_save_keypoints(word, word_id, path):
        print(f"Mock save_keypoints called for {word}, {word_id}, {path}")

    monkeypatch.setattr("ml.features.pipelines.save_keypoints", fake_save_keypoints)
    response = client.get("/save_samples/testword/abc123")
    assert response.status_code == 200
    assert b"testword" in response.data


def test_404_route(client):
    """
    Verifica que acceder a una ruta inexistente retorna un error 404.

    Returns:
        None
    """
    response = client.get("/ruta_inexistente")
    assert response.status_code == 404
