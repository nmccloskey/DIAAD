from __future__ import annotations

from pathlib import Path

import yaml


def write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def sample_target_vocab_resource(resource_id: str = "StoryA") -> dict:
    return {
        "resource_id": resource_id,
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
