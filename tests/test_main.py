import os
import subprocess
import sys
from pathlib import Path

from home_library.main import hello


# Ensure the package can be imported from the src/ layout during tests
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


def test_hello_function_returns_expected_string():
    assert hello() == "Hello, World!"


def test_running_module_prints_hello_world():
    # Run `python -m home_library.main` and verify its stdout
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_PATH)
    proc = subprocess.run(
        [sys.executable, "-m", "home_library.main"],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    assert proc.stdout.strip() == "Hello, World!"
    assert proc.returncode == 0


def test_root_main_script_prints_expected():
    # Execute the repository's root-level main.py script directly
    script_path = PROJECT_ROOT / "main.py"
    assert script_path.exists(), "Root-level main.py should exist"

    proc = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.stdout.strip() == "Hello from home-library!"
    assert proc.returncode == 0
