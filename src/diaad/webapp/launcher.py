from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


def app_path() -> Path:
    """Return the installed Streamlit app script path."""
    return Path(__file__).with_name("streamlit_app.py").resolve()


def _streamlit_available() -> bool:
    return importlib.util.find_spec("streamlit") is not None


def launch_streamlit() -> int:
    """Launch DIAAD's local Streamlit app using the active Python environment."""
    if not _streamlit_available():
        print(
            'DIAAD Streamlit support is not installed. Install it with '
            'pip install "diaad[web]" and then run diaad streamlit.',
            file=sys.stderr,
        )
        return 1

    completed = subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app_path())],
        check=False,
    )
    return completed.returncode


def main() -> int:
    return launch_streamlit()
