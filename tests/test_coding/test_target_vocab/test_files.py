from __future__ import annotations

import json

from diaad.coding.target_vocab import files
from ...helpers import sample_target_vocab_resource


def test_build_target_vocab_template_uses_builtin_shape(monkeypatch):
    monkeypatch.setattr(
        files,
        "load_builtin_resources",
        lambda: {"StoryA": sample_target_vocab_resource("StoryA")},
    )

    template = files.build_target_vocab_template()

    assert template["variant_map"] == {}
    assert set(template["norms"]) == {"accuracy", "efficiency"}


def test_make_target_vocab_file_writes_json(monkeypatch, tmp_path):
    monkeypatch.setattr(
        files,
        "build_target_vocab_template",
        lambda: {"id": "", "display_name": "", "language": "", "task_type": "", "base_forms": [], "variant_map": {}, "norms": {}},
    )

    path = files.make_target_vocab_file(input_dir=tmp_path / "input", output_dir=tmp_path)

    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8"))["id"] == ""
