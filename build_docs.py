import os
import subprocess
import sys
import webbrowser

DOCS_DIR = "docs"
BUILD_HTML = os.path.join(DOCS_DIR, "build", "html", "index.html")


def run_command(command, cwd=None, stop_on_error=True):
    result = subprocess.run(command, shell=True, cwd=cwd)
    if result.returncode != 0:
        print(f"❌ Error al ejecutar: {command}")
        if stop_on_error:
            sys.exit(result.returncode)
    return result.returncode == 0


if __name__ == "__main__":
    print("🧹 Limpiando documentación anterior...")
    run_command("make clean", cwd=DOCS_DIR)

    print("🛠️  Compilando documentación HTML...")
    if run_command("make html", cwd=DOCS_DIR):
        if os.path.exists(BUILD_HTML):
            print("✅ Documentación compilada con éxito.")
            print("🌐 Abriendo en el navegador...")
            webbrowser.open(f"file://{os.path.abspath(BUILD_HTML)}")
        else:
            print("❌ No se encontró index.html.")
    else:
        print("❌ Falló la compilación de la documentación.")
