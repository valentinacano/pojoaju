Instalación
===========

Este proyecto requiere **Python 3.10** y algunas dependencias específicas para funcionar correctamente. A continuación se detallan los pasos necesarios para configurar el entorno de desarrollo e instalar todas las dependencias.

Requisitos previos
------------------

Antes de comenzar, asegurate de tener instalado:

- Homebrew (en macOS)
- pipenv (para manejo de entornos virtuales)
- Python 3.10 (instalado desde Homebrew o manualmente)
- HDF5 (necesario para la librería ``tables``)

Instalación paso a paso
-----------------------

1. Instalar Python 3.10 y HDF5 (en sistemas con Homebrew):

   .. code-block:: bash

      brew install python@3.10
      brew install hdf5

2. Crear y activar un entorno virtual con ``pipenv``:

   .. code-block:: bash

      pipenv --python 3.10
      pipenv shell

3. Instalar la dependencia ``tables`` con soporte binario:

   .. code-block:: bash

      pip install tables==3.10 --only-binary :all:

4. Instalar el resto de las dependencias:

   .. code-block:: bash

      pip install -r requirements.txt

Documentación del proyecto
--------------------------

Para compilar y visualizar esta documentación localmente:

1. Instalar Sphinx y generar el proyecto base:

   .. code-block:: bash

      pip install sphinx
      sphinx-quickstart docs

2. Generar la documentación automáticamente desde el código fuente:

   .. code-block:: bash

      sphinx-apidoc -o docs/source app ml

3. Compilar la documentación:

   .. code-block:: bash

      make html
      open docs/build/html/index.html

   También podés usar un script:

   .. code-block:: bash

      python build_docs.py

Estructura del proyecto
-----------------------

El proyecto sigue una estructura modular para separar claramente la lógica de negocio, la interfaz, los modelos de machine learning y los datos.

Para más información, consultá la sección de estructura de carpetas en el ``README.md``.
