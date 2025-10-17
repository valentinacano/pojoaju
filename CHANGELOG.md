# Changelog

Todas las versiones importantes de Pojoaju se documentan en este archivo.

Este changelog sigue una variante simplificada de [Keep a Changelog](https://keepachangelog.com/es-ES/1.0.0/), con foco en etapas funcionales del proyecto.

---

## [0.2.0] - 2025-08-23

### Agregado
- Botón de "Atrás" en el módulo de *training selector*.
- Botón de "Atrás" en los módulos de subir video y capturar muestras.
- Íconos faltantes en la sección de diccionario.

### Cambios
- Se corrigió el botón "Finalizar entrenamiento" por "Finalizar captura" en la vista de captura de muestras.
- Se actualizaron los íconos en la UI de subir y grabar videos.
- Se adaptó la interfaz de captura de muestras al diseño visual del proyecto.
- Se adaptó la interfaz de subida de videos al diseño visual del proyecto.

---

## [0.2.0] - 2025-07-09

### Agregado
- Implementación inicial de la función `capture_samples_from_video()` para procesar archivos de video y generar muestras desde frames válidos.
- Nueva ruta `GET /training/upload_video/<word_id>/<word>` con formulario para subir videos existentes.
- Nueva ruta `POST /training/upload_video/process/<word_id>/<word>` que guarda el video y ejecuta el pipeline.
- Guardado de archivos con timestamp para evitar sobreescrituras.
- Organización de videos por palabra (`VIDEO_EXPORT_PATH/word/`).
- Sistema de validación y feedback con `flash()` para errores y confirmaciones.
- Test automatizado `test_capture_samples_from_video_crea_muestras_mock` con video artificial y mocks de detección MediaPipe.
- Sección de predicción implementada previamente (modelo LSTM + integración con cámara y texto a voz).

### Cambios
- Se actualizó `capture_samples_from_video()` para guardar muestras solo si se detecta una transición de “mano presente” a “mano ausente”.
- Se corrigió un `BuildError` en Jinja (`url_for`) pasando correctamente los parámetros `word_id` y `word`.
- Se refactorizó la vista `training_selector` para aceptar parámetros y renderizar opciones según la palabra.
- Se mejoró la robustez del procesamiento de videos subidos (verificación de extensión, creación de carpetas, control de flujo).

---

## [0.1.1] - 2025-06-30

### Agregado
- Implementación de la primera versión del pipeline de predicción:
  - `predict_model_from_camera()` con captura desde cámara en tiempo real, integración con `MediaPipe`, detección de manos, predicción con modelo entrenado y síntesis de voz con `text_to_speech`.
- Función `normalize_keypoints()` para interpolar secuencias a una longitud fija compatible con el modelo LSTM.
- Definición del modelo `get_model()` con arquitectura LSTM secuencial:
  - Dos capas LSTM (64 y 128 unidades),
  - Regularización L2,
  - Capas `Dense` intermedias y salida softmax.
- Pipeline de entrenamiento con `training_model()`:
  - Recuperación de secuencias desde base de datos,
  - Preprocesamiento (`pad_sequences`),
  - División entrenamiento/validación,
  - Entrenamiento con `EarlyStopping`,
  - Guardado del modelo final (`MODEL_PATH`).

### Cambios
- Se agregó soporte en `common_utils`, `keypoints_utils` y `database_utils` para:
  - Buscar palabras desde `word_id`.
  - Extraer secuencias de keypoints por palabra.
- Se ajustaron parámetros como `MODEL_FRAMES` y `LENGTH_KEYPOINTS` para una configuración flexible del modelo.

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
