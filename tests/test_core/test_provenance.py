from __future__ import annotations

from argparse import Namespace
from datetime import datetime
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
