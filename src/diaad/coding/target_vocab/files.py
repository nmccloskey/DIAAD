from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from psair.core.logger import get_rel_path, logger

from diaad.coding.target_vocab.resources import load_builtin_resources


TARGET_VOCAB_TEMPLATE_FILENAME = "target_vocabulary_resource_template.json"


def _blank_norm_template() -> dict[str, Any]:
    return {
        "url": "",
        "format": "",
        "columns": {
            "raw_score": "",
            "group": "",
            "pwa_percentile": "",
            "control_percentile": "",
        },
    }


def build_target_vocab_template() -> dict[str, Any]:
    """
    Build a blank target-vocabulary resource template using bundled resource shape.
    """
    resources = load_builtin_resources()
    sample_resource = next(iter(resources.values()))

    return {
        "id": "",
        "display_name": "",
        "language": "",
        "task_type": "",
        "base_forms": [],
        "variant_map": {},
        "norms": {
            norm_name: _blank_norm_template()
            for norm_name in sample_resource.get("norms", {})
        },
    }


def make_target_vocab_file(
    *,
    input_dir: str | Path,
    output_dir: str | Path,
) -> Path:
    """
    Write a blank JSON template for a custom target vocabulary resource.
    """
    del input_dir

    target_vocab_dir = Path(output_dir) / "target_vocab"
    target_vocab_dir.mkdir(parents=True, exist_ok=True)
    output_path = target_vocab_dir / TARGET_VOCAB_TEMPLATE_FILENAME

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(build_target_vocab_template(), f, indent=2)
        f.write("\n")

    logger.info("Wrote target vocabulary template: %s", get_rel_path(output_path))
    return output_path
