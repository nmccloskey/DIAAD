from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from psair.core.logger import get_rel_path, logger

from diaad.coding.target_vocab.resources import (
    get_builtin_resource_ids,
    load_builtin_resources,
    load_resources_from_path,
    load_target_vocabulary_resources,
)


TARGET_VOCAB_TEMPLATE_FILENAME = "target_vocabulary_resource_template.json"
TARGET_VOCAB_CHECK_REPORT_FILENAME = "target_vocab_resource_check.txt"


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


def _format_resource_check_report(
    *,
    resource_path: str | Path | None,
    resources: dict[str, dict[str, Any]],
    custom_resources: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Build the plain-text resource validation report."""
    custom_resources = custom_resources or {}
    lines = ["Target vocabulary resource check", ""]

    if resource_path is None or str(resource_path).strip() == "":
        lines.extend(
            [
                "Custom resource path: none",
                "Custom resource id: none",
            ]
        )
    else:
        resource_display = Path(resource_path)
        if "input" in resource_display.parts:
            resource_display = Path(*resource_display.parts[resource_display.parts.index("input") :])
        lines.append(f"Custom resource path: {resource_display.as_posix()}")
        if len(custom_resources) == 1:
            custom_id = next(iter(custom_resources))
            lines.append(f"Custom resource id: {custom_id}")
        else:
            lines.append(f"Custom resource count: {len(custom_resources)}")
            if custom_resources:
                lines.append("Custom resource ids:")
                lines.extend(f"- {resource_id}" for resource_id in sorted(custom_resources))

    lines.extend(
        [
            f"Active resource count: {len(resources)}",
            "Active resource ids:",
            *[f"- {resource_id}" for resource_id in sorted(resources)],
            "",
            "Built-in narrative resources remain available when a custom JSON path is configured.",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_resource_check_report(
    *,
    output_dir: str | Path,
    report_text: str,
) -> Path:
    target_vocab_dir = Path(output_dir) / "target_vocab"
    target_vocab_dir.mkdir(parents=True, exist_ok=True)
    report_path = target_vocab_dir / TARGET_VOCAB_CHECK_REPORT_FILENAME
    report_path.write_text(report_text, encoding="utf-8", newline="\n")
    logger.info("Wrote target vocabulary resource check report: %s", get_rel_path(report_path))
    return report_path


def check_target_vocab_resources(
    *,
    resource_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Validate and summarize the active target vocabulary resource set.
    """
    if resource_path is None or str(resource_path).strip() == "":
        resources = load_builtin_resources()
        logger.info(
            "No custom target vocabulary resource path configured; validated %d bundled resources.",
            len(resources),
        )
        logger.info("Bundled target vocabulary ids: %s", sorted(resources))
        if output_dir is not None:
            _write_resource_check_report(
                output_dir=output_dir,
                report_text=_format_resource_check_report(
                    resource_path=resource_path,
                    resources=resources,
                ),
            )
        return resources

    custom_resources = load_resources_from_path(resource_path)
    merged_resources = load_target_vocabulary_resources(resource_path)
    builtin_ids = get_builtin_resource_ids()
    override_ids = sorted(set(custom_resources) & builtin_ids)
    custom_only_ids = sorted(set(custom_resources) - builtin_ids)

    logger.info("Validated %d custom target vocabulary resource(s).", len(custom_resources))
    logger.info("Custom target vocabulary ids: %s", sorted(custom_resources))
    if override_ids:
        logger.warning(
            "Custom target vocabulary resources override bundled defaults for ids: %s",
            override_ids,
        )
    if custom_only_ids:
        logger.info(
            "Custom target vocabulary resources add non-bundled ids: %s",
            custom_only_ids,
        )
    logger.info("Active target vocabulary ids after merge: %s", sorted(merged_resources))
    if output_dir is not None:
        _write_resource_check_report(
            output_dir=output_dir,
            report_text=_format_resource_check_report(
                resource_path=resource_path,
                resources=merged_resources,
                custom_resources=custom_resources,
            ),
        )
    return merged_resources
