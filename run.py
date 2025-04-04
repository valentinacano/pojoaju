import subprocess
import sys


def run_command(command, stop_on_error=False):
    result = subprocess.run(command, shell=True)
    if result.returncode != 0 and stop_on_error:
        print(f"\n❌ Error al ejecutar: {command}")
        sys.exit(result.returncode)


if __name__ == "__main__":
    print("✨ Ejecutando Black (formateo)...")
    run_command("black .")

    print("🔍 Ejecutando Pyflakes (chequeo de errores)...")
    result = subprocess.run("pyflakes .", shell=True, capture_output=True, text=True)

    if result.stdout:
        print("\n🚫 Pyflakes detectó errores:")
        print(result.stdout)
        sys.exit(1)
    else:
        print("✅ Pyflakes no encontró errores. Ejecutando main.py...\n")
        run_command("python main.py")
