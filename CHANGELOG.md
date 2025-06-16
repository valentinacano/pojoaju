# Changelog

Todas las versiones importantes de Pojoaju se documentan en este archivo.

Este changelog sigue una variante simplificada de [Keep a Changelog](https://keepachangelog.com/es-ES/1.0.0/), con foco en etapas funcionales del proyecto.

---

## [0.1.0] - 2025-06-16

### Agregado
- Conexión a base de datos PostgreSQL mediante `get_connection()`.
- Scripts para crear las tablas: `categories`, `words`, `samples`, `keypoints`.
- Inserción de palabras por categoría y muestras con keypoints.
- Captura de keypoints con MediaPipe Holistic.
- Entrenamiento inicial del modelo con palabras del alfabeto, números, colores, emociones y saludos.
- Infraestructura de testing: `pytest`, fixture de limpieza de base y prueba de conexión.
- Configuración de entorno con Pipenv.
- Instalación editable del proyecto con `setup.py`.

### Documentación
- Documentación técnica inicial en `README.md` con:
  - Estructura de carpetas.
  - Instrucciones de instalación y testing.
  - Justificación de `pip install -e .`.
  - Buenas prácticas de versionado.
- Setup de documentación automática con Sphinx (`build_docs.py`, `docs/`).

### Cambios
- Estructura modular inicial del proyecto organizada en `app/`, `ml/`, `tests/`, `data/`, `docs/`, etc.
