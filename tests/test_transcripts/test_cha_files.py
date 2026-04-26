from __future__ import annotations

from pathlib import Path

import diaad.transcripts.cha_files as cha_files


def test_truncate_path_to_input_dir_returns_portable_relative_path(tmp_path):
    input_dir = tmp_path / "input"
    file_path = input_dir / "sub" / "sample.cha"
    file_path.parent.mkdir(parents=True)
    file_path.write_text("", encoding="utf-8")

    result = cha_files.truncate_path_to_input_dir(file_path, input_dir)

    assert result == "input/sub/sample.cha"


def test_read_cha_files_uses_input_relative_keys(monkeypatch, tmp_path):
    input_dir = tmp_path / "input"
    file_path = input_dir / "sample.cha"
    file_path.parent.mkdir(parents=True)
    file_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(cha_files.pylangacq, "read_chat", lambda path: {"path": path})
    monkeypatch.setattr(cha_files, "tqdm", lambda items, **kwargs: items)

    chats = cha_files.read_cha_files(input_dir)

    assert chats["input/sample.cha"]["path"] == str(file_path)
