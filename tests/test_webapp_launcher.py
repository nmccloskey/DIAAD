from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from diaad.webapp import launcher


def test_launcher_runs_streamlit_against_packaged_app(monkeypatch):
    calls = []

    def fake_run(command, *, check):
        calls.append((command, check))
        return SimpleNamespace(returncode=23)

    monkeypatch.setattr(launcher, "_streamlit_available", lambda: True)
    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    monkeypatch.setattr(launcher.sys, "executable", "python")

    assert launcher.main() == 23

    command, check = calls[0]
    assert command == [
        "python",
        "-m",
        "streamlit",
        "run",
        str(launcher.app_path()),
    ]
    assert check is False
    assert Path(command[-1]).name == "streamlit_app.py"


def test_launcher_prints_web_extra_hint_when_streamlit_is_missing(monkeypatch, capsys):
    monkeypatch.setattr(launcher, "_streamlit_available", lambda: False)

    assert launcher.main() == 1

    captured = capsys.readouterr()
    assert 'pip install "diaad[web]"' in captured.err
    assert "diaad streamlit" in captured.err
