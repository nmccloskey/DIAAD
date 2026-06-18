from __future__ import annotations

from argparse import Namespace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from diaad.core import provenance


def _fake_ctx(tmp_path):
    config_source = {
        "kind": "nested_file",
        "path": str(tmp_path / "config.yaml"),
        "files": {"config": str(tmp_path / "config.yaml")},
        "missing_sections": [],
        "defaults_applied": True,
        "default_path": str(tmp_path / "default_config.yaml"),
    }
    config = SimpleNamespace(
        override_diff={},
        config_source=config_source,
        to_dict=lambda: {"project": {"input_dir": "input"}, "advanced": {}},
    )
    return SimpleNamespace(
        config=config,
        commands=["powers evaluate"],
        timestamp="260520_1237",
        run_paths=lambda: {"config_source": config_source["path"]},
    )


def test_build_dry_run_payload_includes_config_source(tmp_path):
    ctx = _fake_ctx(tmp_path)
    args = Namespace(command=["powers", "evaluate"], config=str(tmp_path / "config.yaml"))

    payload = provenance.build_dry_run_payload(ctx, args, ["powers evaluate"])

    assert payload["config_source"] == ctx.config.config_source
    assert payload["paths"]["config_source"] == ctx.config.config_source["path"]


def test_compact_run_metadata_includes_config_source(tmp_path):
    ctx = _fake_ctx(tmp_path)

    metadata = provenance._compact_run_metadata(
        ctx=ctx,
        status="completed",
        start_time=datetime(2026, 5, 20, 12, 37),
        end_time=datetime(2026, 5, 20, 12, 38),
    )

    assert metadata["config_source"] == ctx.config.config_source
    assert metadata["paths"]["config_source"] == ctx.config.config_source["path"]


def test_run_directory_snapshot_uses_long_path_safe_roots(monkeypatch, tmp_path):
    calls = []
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    ctx = SimpleNamespace(
        input_dir=input_dir,
        out_dir=output_dir,
        project_root=tmp_path,
    )

    def fake_long_path(path):
        calls.append(path)
        return f"long::{Path(path).name}"

    def fake_capture_directory_snapshot(base, *, root, include_file_stats=True):
        return {
            "base": base,
            "root": root,
            "include_file_stats": include_file_stats,
        }

    monkeypatch.setattr(provenance, "long_path", fake_long_path)
    monkeypatch.setattr(
        provenance,
        "capture_directory_snapshot",
        fake_capture_directory_snapshot,
    )

    snapshot = provenance._run_directory_snapshot(ctx)

    assert calls == [input_dir, tmp_path, output_dir, output_dir]
    assert snapshot["input_contents"] == {
        "base": "long::input",
        "root": f"long::{tmp_path.name}",
        "include_file_stats": True,
    }
    assert snapshot["output_contents"] == {
        "base": "long::output",
        "root": "long::output",
        "include_file_stats": True,
    }


def test_capture_directory_snapshot_falls_back_without_stats(monkeypatch, tmp_path):
    attempts = []

    def fake_capture_directory_snapshot(base, *, root, include_file_stats=True):
        attempts.append(include_file_stats)
        if include_file_stats:
            raise FileNotFoundError("path vanished during stat")
        return {"base": str(base), "root": str(root), "files": ["out.xlsx"]}

    monkeypatch.setattr(
        provenance,
        "capture_directory_snapshot",
        fake_capture_directory_snapshot,
    )

    snapshot = provenance._capture_directory_snapshot(tmp_path, root=tmp_path)

    assert attempts == [True, False]
    assert snapshot["files"] == ["out.xlsx"]
    assert "path vanished during stat" in snapshot["snapshot_warning"]
