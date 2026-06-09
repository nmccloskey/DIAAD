from __future__ import annotations

from copy import deepcopy
from collections.abc import Iterable, Mapping
from typing import Any

from psair.core.provenance import ParsedOverride, parse_key_value_overrides


PROJECT_KEYS = {
    "input_dir",
    "output_dir",
    "random_seed",
    "reliability_fraction",
    "shuffle_samples",
    "strip_clan",
    "prefer_correction",
    "lowercase",
    "exclude_speakers",
    "auto_tabularize",
    "num_bins",
    "num_coders",
    "stimulus_column",
    "automate_powers",
    "metadata_fields",
}

ADVANCED_KEYS = {
    "transcript_table_filename",
    "sample_id_column",
    "utterance_id_column",
    "reliability_tag",
    "reliability_dirname",
    "cu_paradigms",
    "cu_samples_filename",
    "cu_utts_filename",
    "word_count_filename",
    "word_count_column",
    "wc_samples_filename",
    "speaking_time_filename",
    "speaking_time_column",
    "target_vocabulary_resource_path",
    "auto_blind",
    "blind_columns",
    "metadata_source",
    "codebook_filename",
    "powers_coding_filename",
    "powers_reliability_filename",
}


def parse_config_overrides(items: Iterable[str] | None) -> dict[str, Any]:
    """Parse and validate DIAAD ``--set KEY=VALUE`` config overrides."""
    parsed = parse_key_value_overrides(items)
    return {_canonical_key(item): item.value for item in parsed}


def build_cli_config_overrides(args) -> dict[str, Any]:
    """Build config overrides from direct CLI flags and repeated ``--set``."""
    overrides = parse_config_overrides(getattr(args, "set_values", None))

    input_dir = getattr(args, "input_dir", None)
    if input_dir is not None:
        overrides["project.input_dir"] = input_dir

    output_dir = getattr(args, "output_dir", None)
    if output_dir is not None:
        overrides["project.output_dir"] = output_dir

    return overrides


def apply_config_overrides(
    project_data: Mapping[str, Any],
    advanced_data: Mapping[str, Any],
    overrides: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return copied project/advanced config dictionaries with overrides applied."""
    project = deepcopy(dict(project_data))
    advanced = deepcopy(dict(advanced_data))

    for key, value in (overrides or {}).items():
        section, field = _split_canonical_key(key)
        if section == "project":
            project[field] = value
        elif section == "advanced":
            advanced[field] = value
        else:
            raise ValueError(f"Unsupported config override section: {section}")

    return project, advanced


def _canonical_key(override: ParsedOverride) -> str:
    key = override.key.strip()
    section, field = _split_key(key)

    if section is None:
        if field in PROJECT_KEYS:
            return f"project.{field}"
        if field in ADVANCED_KEYS:
            return f"advanced.{field}"
        raise ValueError(f"Unknown DIAAD config override key: {key}")

    if section == "project" and field in PROJECT_KEYS:
        return f"project.{field}"
    if section == "advanced" and field in ADVANCED_KEYS:
        return f"advanced.{field}"

    raise ValueError(f"Unknown DIAAD config override key: {key}")


def _split_key(key: str) -> tuple[str | None, str]:
    if "." not in key:
        return None, key
    section, field = key.split(".", 1)
    if not section or not field or "." in field:
        raise ValueError(f"Unsupported DIAAD config override key: {key}")
    if section not in {"project", "advanced"}:
        raise ValueError(f"Unsupported DIAAD config override section: {section}")
    return section, field


def _split_canonical_key(key: str) -> tuple[str, str]:
    section, field = _split_key(key)
    if section is None:
        raise ValueError(f"Expected canonical config key with section: {key}")
    return section, field
