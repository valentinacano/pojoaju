from setuptools import setup, find_packages

setup(
    name="pojoaju",
    version="0.1.0",
    packages=find_packages(),  # Detecta automáticamente app/, ml/, etc.
    include_package_data=True,
    install_requires=[],  # Pipenv ya gestiona las dependencias
    description="Proyecto de reconocimiento de lengua de señas paraguaya mediante redes neuronales",
    author="Valentina Cano y Sergio Espínola",
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
