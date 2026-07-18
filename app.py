import os
import runpy
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(ROOT_DIR, "backend")

if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

if __name__ == "__main__":
    os.chdir(BACKEND_DIR)
    runpy.run_path(os.path.join(BACKEND_DIR, "app.py"), run_name="__main__")
