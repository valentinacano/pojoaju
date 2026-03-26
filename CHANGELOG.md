# Changelog

Todas las versiones importantes de Pojoaju se documentan en este archivo.

---

## [1.0.0] - 24.03.2026

### Refactor completo — proyecto rehecho desde cero

#### Problemas críticos resueltos
- **Normalización unificada**: `normalize_sequence()` es ahora el único método usado en training, predicción y evaluación. Antes había tres implementaciones distintas (interpolación, recorte, pad_sequences), lo que causaba baja accuracy.
- **`test_size` corregido**: de 0.05 → 0.2 en training (con estratificación). El valor anterior dejaba 0-1 muestras de validación por clase.
- **Labels de matriz de confusión**: ahora muestran nombres reales de palabras. Antes mostraban direcciones de memoria de Python.
- **`test_size` en evaluación**: corregido de 0.8 → 0.3 (era un bug — el modelo se evaluaba con el 80% de los datos y entrenaba con el 20%).
- **`text_to_speech` unificada**: eliminada la implementación duplicada en `predict_model_from_camera.py`.

#### Arquitectura
- Estructura simplificada: `ml/utils/`, `ml/features/`, `ml/prediction/`, `ml/training/` → aplanado a `ml/*.py`
- `app/database/database_utils.py` → `app/database/queries.py` (todo el SQL en un solo lugar)
- `ml/pipeline.py` reemplaza `ml/features/pipelines.py`

#### Modelo
- Dropout reducido: 0.5 → 0.2 (era demasiado agresivo para datasets pequeños)
- L2 uniforme en todas las capas (antes era 10x más fuerte en la primera)
- `BatchNormalization` agregado para estabilizar el entrenamiento
- `EarlyStopping(patience=30)` y `ModelCheckpoint` para guardar el mejor modelo
- `batch_size` aumentado: 8 → 16

#### Archivos eliminados
- `ml/utils/common_utils.py`
- `ml/utils/keypoints_utils.py`
- `ml/utils/normalize_utils.py`
- `ml/utils/capture_utils.py`
- `ml/utils/training_utils.py`
- `ml/utils/visualize_utils.py`
- `ml/features/pipelines.py`
- `ml/features/capture_samples.py`
- `ml/features/capture_samples_video.py`
- `ml/features/normalize_samples.py`
- `ml/features/create_keypoints.py`
- `ml/features/visualizer.py`
- `ml/prediction/predict_model_from_camera.py`
- `ml/training/confusion_utils.py`
- `app/database/database_utils.py`

---

## [0.2.0] - 2025-08-23

### Agregado
- Botón de "Atrás" en el módulo de *training selector*.
- Botón de "Atrás" en los módulos de subir video y capturar muestras.
- Íconos faltantes en la sección de diccionario.

### Cambios
- Se corrigió el botón "Finalizar entrenamiento" por "Finalizar captura".
- Se actualizaron los íconos en la UI.
- Se adaptó la interfaz de captura y subida de videos al diseño visual.

---

## [0.2.0] - 2025-07-09

### Agregado
- `capture_samples_from_video()` para procesar archivos de video.
- Rutas para subir videos existentes.
- Sistema de validación con `flash()`.

---

## [0.1.1] - 2025-06-30

### Agregado
- Primera versión del pipeline de predicción.
- `normalize_keypoints()` para interpolación de secuencias.
- Modelo `get_model()` con arquitectura LSTM.
- Pipeline de entrenamiento con `EarlyStopping`.

---

## [0.1.0] - 2025-06-16

### Agregado
- Conexión a PostgreSQL.
- Tablas: `categories`, `words`, `samples`, `keypoints`.
- Captura de keypoints con MediaPipe Holistic.
- Infraestructura de testing con pytest.
- Configuración con Pipenv y `setup.py`.