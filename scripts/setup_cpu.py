"""Create an isolated Python 3.11 CPU runtime; never remove research caches."""
import argparse
from pathlib import Path
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parents[1]


def create_environment(destination):
    destination = Path(destination).resolve()
    venv.EnvBuilder(with_pip=True).create(destination)
    python = destination / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    subprocess.run([str(python), "-m", "pip", "install", "-r", str(ROOT / "requirements-cpu.txt")], check=True)
    return python


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", type=Path, default=ROOT / ".venv")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 11):
        parser.error("Use Python 3.11 to create the tested environment")
    python = create_environment(args.venv)
    print(f"Environment ready. Run: {python} -m examples.cpu_demo")


if __name__ == "__main__":
    main()
