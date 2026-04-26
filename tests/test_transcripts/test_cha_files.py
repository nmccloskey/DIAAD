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

    def fake_from_files(paths, *, parallel=True):
        return {"path": paths[0], "parallel": parallel}

    monkeypatch.setattr(
        cha_files.pylangacq.Reader,
        "from_files",
        staticmethod(fake_from_files),
    )
    monkeypatch.setattr(cha_files, "tqdm", lambda items, **kwargs: items)

    chats = cha_files.read_cha_files(input_dir)

    assert chats["input/sample.cha"]["path"] == str(file_path)
    assert chats["input/sample.cha"]["parallel"] is False


def test_read_cha_files_can_skip_excluded_directories(monkeypatch, tmp_path):
    input_dir = tmp_path / "input"
    original = input_dir / "sample.cha"
    reliability = input_dir / "reliability" / "sample.cha"
    original.parent.mkdir(parents=True)
    reliability.parent.mkdir(parents=True)
    original.write_text("", encoding="utf-8")
    reliability.write_text("", encoding="utf-8")

    monkeypatch.setattr(
        cha_files.pylangacq.Reader,
        "from_files",
        staticmethod(lambda paths, *, parallel=True: {"path": paths[0]}),
    )
    monkeypatch.setattr(cha_files, "tqdm", lambda items, **kwargs: items)

    chats = cha_files.read_cha_files(input_dir, exclude_dirnames=["reliability"])

    assert list(chats) == ["input/sample.cha"]
