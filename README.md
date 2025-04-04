Este proyecto requiere Python 3.10 y algunas dependencias específicas. A continuación se detallan los pasos para configurar el entorno y la estructura de carpetas sugerida.

## Setup del entorno

1. Asegurate de tener Homebrew instalado.  
2. Instalá Python 3.10:  
   ```bash
   brew install python@3.10
   ```
3. Instalá hdf5:  
   ```bash
   brew install hdf5
   ```
4. Creá el entorno virtual con Pipenv:  
   ```bash
   pipenv --python 3.10
   ```
5. Activá el entorno virtual:  
   ```bash
   pipenv shell
   ```
6. Instalá el paquete tables desde binarios:  
   ```bash
   pip install tables==3.10 --only-binary :all:
   ```
7. Instalá los requerimientos del proyecto:  
   ```bash
   pip install -r requirements.txt
   ```

## Documentación

1. Instalá sphinx:
   ```bash
   pip install sphinx
   ```
2. Configuración del entorno:
   ```bash
   sphinx-quickstart docs
   ```
3. Crear/actualizar documentos:
   ```bash
   sphinx-apidoc -o docs/source app ml
   ```
4. Compilar archivos html:
   ```bash
   make html
   ```
5. Abrir documentación: 
   ```bash
   open docs/build/html/index.html
   ```
o simplemente ejercutar:
```bash
   open docs/build/html/index.html
   ```


## Estructura de carpetas propuesta

```plaintext
pojoaju/
├── app/                   
│   ├── models/             # Clases y lógica de datos (entidades, estructuras)
│   ├── views/              # Presentación (consola, HTML, UI)
│   ├── services/           # Lógica de negocio y procesamiento
│   ├── utils/              # Funciones auxiliares o genéricas
│   ├── main.py             # Punto de entrada
│   └── config.py           # Configuraciones generales (paths, parámetros)
├── ml/                     
│   ├── training/           # Scripts de entrenamiento
│   ├── prediction/         # Scripts de predicción
│   ├── features/           # Ingeniería de features
│   └── models/             # Modelos ya entrenados (.pkl, .joblib, etc.)
├── notebooks/              # Jupyter Notebooks para exploración y pruebas
├── data/                   
│   ├── raw/                # Datos crudos
│   ├── processed/          # Datos procesados / limpios
│   └── external/           # Datos externos (de terceros)
├── db/                     # Conexión, esquemas y lógica de acceso a BD
├── tests/                  # Pruebas del proyecto
├── static/                 
│   ├── html/
│   └── css/
├── requirements.txt        # Dependencias
├── instalacion.sh          # Script de instalación automática
└── README.md               # Instrucciones del proyecto
```

| Carpeta             | Descripción                                                                 |
|---------------------|------------------------------------------------------------------------------|
| `app/`              | Código base de la aplicación. Contiene la lógica general y configuración.    |
| `app/models/`       | Clases y estructuras de datos (por ejemplo, clases tipo `User`, `Record`).   |
| `app/views/`        | Visualización o presentación (CLI, HTML, dashboards, si aplica).             |
| `app/services/`     | Lógica de negocio: procesos, validaciones, transformaciones generales.       |
| `app/utils/`        | Funciones auxiliares reutilizables (logs, fechas, formateos, etc.).          |
| `app/main.py`       | Punto de entrada del proyecto. Junta todo lo necesario para ejecutar.        |
| `app/config.py`     | Parámetros globales del proyecto (paths, flags, hiperparámetros, etc.).      |
| `ml/`               | Código específico de machine learning.                                       |
| `ml/training/`      | Scripts para entrenamiento de modelos.                                       |
| `ml/prediction/`    | Scripts para hacer predicciones con modelos entrenados.                      |
| `ml/features/`      | Scripts de ingeniería de features: transformación y selección de variables.  |
| `ml/models/`        | Modelos entrenados guardados (.pkl, .joblib, etc.).                          |
| `notebooks/`        | Notebooks Jupyter para análisis exploratorio y pruebas.                      |
| `data/`             | Datos utilizados en el proyecto.                                             |
| `data/raw/`         | Datos originales sin procesar.                                               |
| `data/processed/`   | Datos limpios, transformados y listos para entrenar.                         |
| `data/external/`    | Datos externos de terceros.                                                  |
| `db/`               | Lógica de conexión, consultas y definiciones de base de datos.               |
| `tests/`            | Pruebas unitarias y de integración para validar el código.                   |
| `static/`           | Recursos estáticos como HTML, CSS, JS (si hay una interfaz visual).          |
| `requirements.txt`  | Lista de dependencias del proyecto.                                          |
| `instalacion.sh`    | Script para automatizar instalación de entorno y dependencias.              |
| `README.md`         | Documentación general del proyecto. 
>>>>>>> develop

## Traducción Incial

1. A - Z
2. a - z
3. 0 - 9
4. Saludos y expresiones básicas
   - Hola
   - Chau
   - Buenos días
   - Buenas tardes
   - Buenas noches
   - Gracias
   - Por favor
   - Perdón
5. Colores
   - Rojo
   - Azul
   - Verde
   - Amarillo
   - Negro
   - Blanco
   - Naranja
6. Animales
   - Perro
   - Gato
   - Vaca
   - Caballo
   - Cerdo
   - Gallina
   - Pájaro
   - Ratón
7. Días de la semana y Meses
8. Tiempo
   - Hoy
   - Ayer
   - Mañana
   - Tarde
   - Noche
   - Hora
   - Minuto
9. Familia y personas
   - Mamá
   - Papá
   - Hermano
   - Hermana
   - Abuela
   - Abuelo
   - Mujer
   - Hombre
10. Básicos
   - Desayuno
   - Almuerzo
   - Cena
   - Baño
   - Comer
   - Tomar
   - Dormir
   - Sueño
   - Hambre
   - Sed
11. Emociones
   - Feliz
   - Triste
   - Enojado/a
   - Asustado/a
   - Cansado/a
   - Llorar
   - Reír
   - Amar