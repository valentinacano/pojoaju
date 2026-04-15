# Changelog

Todas las versiones importantes de Pojoaju se documentan en este archivo.

---

## [1.1.0] - 2026-03-26

### Agregado
- **Texto a Señas**: nueva sección que permite escribir una palabra y ver su seña animada con un stickman. Usa los keypoints promedio almacenados en la BD y los anima frame por frame en un canvas HTML.
- **Voz a Señas**: nueva sección que usa la Web Speech API del navegador para reconocer palabras en tiempo real (español Paraguay) y mostrar la seña correspondiente automáticamente.
- `ml/sign_animator.py`: módulo que extrae la muestra más representativa de la BD, aplica interpolación entre frames para suavizar la animación, y genera el JSON de coordenadas para el frontend.
- Ruta `/text_to_sign` y `/voice_to_sign` en Flask.
- Ruta `/api/sign/<word>` que retorna la animación en JSON.
- Leyenda de manos (izquierda/derecha) en el canvas del stickman.

### Cambios
- `index.html`: botones "Texto a Señas" y "Voz a Señas" ahora enlazan a sus respectivas rutas.

---

## [1.0.0] - 2026-03-26

### Refactor completo — proyecto rehecho desde cero

#### Problemas críticos resueltos
- **Normalización unificada**: `normalize_sequence()` es ahora el único método usado en training, predicción y evaluación. Antes había tres implementaciones distintas, lo que causaba baja accuracy.
- **`test_size` corregido**: de 0.05 → 0.2 en training con estratificación.
- **Labels de matriz de confusión**: ahora muestran nombres reales de palabras.
- **`test_size` en evaluación**: corregido de 0.8 → 0.3.
- **`text_to_speech` unificada**: eliminada la implementación duplicada.
- **`_graph is None` en MediaPipe**: Holistic ahora se mantiene abierto durante todo el generador de captura.
- **Validación numpy**: corregido `if not sequence` → `if sequence is None or len(sequence) == 0`.
- **Muestras con distinto número de frames**: normalización antes de comparar en `_get_best_sample`.

#### Arquitectura
- Estructura simplificada: `ml/utils/`, `ml/features/`, `ml/prediction/`, `ml/training/` → aplanado a `ml/*.py`
- `app/database/database_utils.py` → `app/database/queries.py`
- `ml/pipeline.py` reemplaza `ml/features/pipelines.py`

#### Modelo
- Dropout reducido: 0.5 → 0.2
- L2 uniforme en todas las capas
- `BatchNormalization` agregado
- `EarlyStopping(patience=30)` y `ModelCheckpoint`
- `batch_size` aumentado: 8 → 16
- `test_size` corregido: 0.05 → 0.2 con estratificación

#### Archivos eliminados
- `ml/utils/` (6 archivos)
- `ml/features/` (7 archivos)
- `ml/prediction/predict_model_from_camera.py`
- `ml/training/confusion_utils.py`, `model.py`, `training_model.py`
- `app/database/database_utils.py`
- `build_docs.py`, `docs/`, `Pipfile.lock`

---

## [0.2.0] - 2025-08-23

### Agregado
- Botón de "Atrás" en training selector, captura y subida de videos.
- Íconos faltantes en la sección de diccionario.

### Cambios
- Corregido el botón "Finalizar entrenamiento" → "Finalizar captura".
- Interfaz de captura y subida de videos adaptada al diseño visual del proyecto.

---

## [0.2.0] - 2025-07-09

### Agregado
- `capture_samples_from_video()` para procesar archivos de video.
- Rutas para subir videos existentes con timestamp para evitar sobreescrituras.
- Sistema de validación con `flash()`.
- Test automatizado para captura desde video con mocks de MediaPipe.

---

## [0.1.1] - 2025-06-30

### Agregado
- Primera versión del pipeline de predicción con cámara en tiempo real.
- `normalize_keypoints()` para interpolación de secuencias.
- Modelo `get_model()` con arquitectura LSTM.
- Pipeline de entrenamiento con `EarlyStopping`.

---

## [0.1.0] - 2025-06-16

### Agregado
- Conexión a PostgreSQL y creación de tablas.
- Captura de keypoints con MediaPipe Holistic.
- Infraestructura de testing con pytest.
- Configuración con Pipenv y `setup.py`.
- Documentación técnica inicial en `README.md`.