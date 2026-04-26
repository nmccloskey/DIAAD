from __future__ import annotations

from pathlib import Path
import shutil
import uuid

import pytest
import yaml


def write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def sample_target_vocab_resource(resource_id: str = "StoryA") -> dict:
    return {
        "id": resource_id,
        "display_name": f"{resource_id} Display",
        "language": "en",
        "task_type": "narrative",
        "base_forms": ["cat", "dog"],
        "variant_map": {
            "cat": ["cats"],
            "dog": ["dogs"],
        },
        "norms": {
            "accuracy": {
                "url": "https://example.com/accuracy.csv",
                "format": "csv",
                "columns": {
                    "raw_score": "score",
                    "group": "group",
                    "pwa_percentile": "pwa_pct",
                    "control_percentile": "ctl_pct",
                },
            },
            "efficiency": {
                "url": "https://example.com/efficiency.csv",
                "format": "csv",
                "columns": {
                    "raw_score": "score",
                    "group": "group",
                    "pwa_percentile": "pwa_pct",
                    "control_percentile": "ctl_pct",
                },
            },
        },
    }


@pytest.fixture
def tmp_path() -> Path:
    base_dir = Path.cwd() / ".tmp_pytest"
    base_dir.mkdir(parents=True, exist_ok=True)
    path = base_dir / f"tmp_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
