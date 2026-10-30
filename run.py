"""
EmoCraft — All-in-One Runner
=============================
Run the entire project with a single command:
    python run.py

This script:
  1. Detects Python environment and validates dependencies.
  2. Verifies that the trained models and face detector exist.
  3. Automatically opens your default web browser to the app URL.
  4. Starts the Streamlit application server.

Additional options:
    python run.py --test            Run the full smoke test suite
    python run.py --train-image     Retrain the image model on FER2013
    python run.py --train-video     Retrain the video model
    python run.py --no-browser      Start the server without opening browser
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent
VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
VENV_STREAMLIT = PROJECT_ROOT / ".venv" / "Scripts" / "streamlit.exe"

# Select best Python executable (prefer .venv if available)
PYTHON_EXE = str(VENV_PYTHON) if VENV_PYTHON.is_file() else sys.executable
STREAMLIT_EXE = str(VENV_STREAMLIT) if VENV_STREAMLIT.is_file() else None


def print_banner() -> None:
    print("=" * 65)
    print("   🎭  EMOCRAFT — ALL-IN-ONE PROJECT RUNNER")
    print("=" * 65)


def check_models() -> bool:
    """Verify that models exist before starting the app."""
    models_dir = PROJECT_ROOT / "models"
    img_model = models_dir / "emocraft_image.keras"
    face_model = models_dir / "face_detection_yunet_2023mar.onnx"

    missing = []
    if not face_model.is_file():
        missing.append("YuNet face detector (models/face_detection_yunet_2023mar.onnx)")
    if not img_model.is_file():
        missing.append("Trained image model (models/emocraft_image.keras)")

    if missing:
        print("\n[WARN] The following model files were not found:")
        for m in missing:
            print(f"  - {m}")
        print("\nRunning smoke test to auto-download prerequisites ...\n")
        subprocess.run([PYTHON_EXE, str(PROJECT_ROOT / "smoke_test.py")])
        if not img_model.is_file():
            print("\n[INFO] Training initial image model ...\n")
            subprocess.run([PYTHON_EXE, str(PROJECT_ROOT / "train_image.py"), "--epochs", "3", "--batch-size", "256"])
    return True


def launch_app(port: int = 8501, open_browser: bool = True) -> None:
    """Launch Streamlit and open web browser."""
    url = f"http://localhost:{port}"
    print(f"\n[1/2] Preparing Streamlit application on {url} ...")
    check_models()

    if open_browser:
        print(f"[2/2] Opening {url} in your default browser ...")
        # Delay opening browser slightly until server binds
        import threading
        def _open():
            time.sleep(1.5)
            webbrowser.open(url)
        threading.Thread(target=_open, daemon=True).start()

    print("\n" + "=" * 65)
    print(f"  🚀  EmoCraft Web App is RUNNING at:")
    print(f"      Local:   {url}")
    print("=" * 65)
    print("  Press Ctrl+C in this terminal to stop the application.\n")

    cmd = [PYTHON_EXE, "-m", "streamlit", "run", str(PROJECT_ROOT / "app.py"),
           "--server.port", str(port), "--server.headless", "true"]
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n\n[INFO] Application stopped. Goodbye!")


def main() -> None:
    parser = argparse.ArgumentParser(description="EmoCraft All-in-One Runner")
    parser.add_argument("--test", action="store_true", help="Run the full smoke test suite")
    parser.add_argument("--train-image", action="store_true", help="Retrain the image model on FER2013")
    parser.add_argument("--train-video", action="store_true", help="Retrain the video model")
    parser.add_argument("--port", type=int, default=8501, help="Port for the web app (default: 8501)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open web browser automatically")
    args = parser.parse_args()

    print_banner()

    if args.test:
        print("[MODE] Running Smoke Test ...\n")
        subprocess.run([PYTHON_EXE, str(PROJECT_ROOT / "smoke_test.py")])
        return

    if args.train_image:
        print("[MODE] Retraining Image Model on FER2013 ...\n")
        subprocess.run([PYTHON_EXE, str(PROJECT_ROOT / "train_image.py")])
        return

    if args.train_video:
        print("[MODE] Retraining Video Model ...\n")
        subprocess.run([PYTHON_EXE, str(PROJECT_ROOT / "train_video.py")])
        return

    # Default action: launch the web app and open the browser
    launch_app(port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
