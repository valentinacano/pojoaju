"""
Tests de rutas y comportamiento de la aplicación Flask.

Este módulo verifica el correcto funcionamiento de las rutas de la interfaz web
incluidas en `app.views.flask_gui`. Se testean rutas de carga de plantillas, 
formularios, flujos de captura y respuestas esperadas del servidor.

Incluye también validaciones de errores comunes, redirecciones y comportamiento 
frente a rutas inexistentes.
"""

import pytest
import app.views.flask_gui as flask_gui  # ✅ Necesario para patching correcto


@pytest.fixture
def client():
    """
    Crea un cliente de prueba para la app Flask.

    Returns:
        flask.testing.FlaskClient: Cliente de prueba.
    """
    flask_gui.app.config["TESTING"] = True
    with flask_gui.app.test_client() as client:
        yield client


def test_index_route(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"<!DOCTYPE html" in response.data


def test_training_route(client):
    response = client.get("/training")
    assert response.status_code == 200
    assert b"<!DOCTYPE html" in response.data


def test_capture_form_route(client):
    response = client.get("/capture_form")
    assert response.status_code == 200
    assert b"<!DOCTYPE html" in response.data


def test_dictionary_route(client):
    response = client.get("/training/dictionary")
    assert response.status_code == 200
    assert b"<!DOCTYPE html" in response.data


def test_insert_word_form_get(client):
    response = client.get("/training/insert_word")
    assert response.status_code == 200
    assert b"<!DOCTYPE html" in response.data


def test_insert_word_form_post_missing_fields(client):
    response = client.post(
        "/training/insert_word",
        data={"word": "", "category_existing": "", "category_new": ""},
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert b"La palabra es obligatoria" in response.data


def test_insert_word_form_post_missing_category(client):
    response = client.post(
        "/training/insert_word",
        data={"word": "hola", "category_existing": "", "category_new": ""},
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert b"Debe seleccionar o ingresar una categor" in response.data


def test_capture_route(client):
    response = client.get("/training/capture/testid/testword")
    assert response.status_code == 200
    assert b"testword" in response.data


def test_stop_capture_without_data(client):
    response = client.post("/stop_capture", data={}, follow_redirects=True)
    assert response.status_code == 200
    assert b"<!DOCTYPE html" in response.data


def test_stop_capture_with_data(client):
    response = client.post(
        "/stop_capture",
        data={"word": "hola", "word_id": "abc123"},
        follow_redirects=False,
    )
    assert response.status_code == 302
    assert "/save_samples/hola/abc123" in response.headers["Location"]


def test_video_feed_route(client):
    response = client.get("/video_feed/test")
    assert response.status_code == 200
    assert response.mimetype == "multipart/x-mixed-replace"


def test_save_samples_route(client, monkeypatch):
    def fake_save_keypoints(word, word_id, path):
        print(f"Mock save_keypoints called for {word}, {word_id}, {path}")

    monkeypatch.setattr("ml.features.pipelines.save_keypoints", fake_save_keypoints)
    response = client.get("/save_samples/testword/abc123")
    assert response.status_code == 200
    assert b"testword" in response.data


def test_404_route(client):
    response = client.get("/ruta_inexistente")
    assert response.status_code == 404
