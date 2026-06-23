"""Resource loading for target vocabulary coverage analysis."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


BUILTIN_RESOURCE_DIR = Path(__file__).with_name("resources")
REQUIRED_RESOURCE_KEYS = {
    "resource_id",
    "display_name",
    "language",
    "task_type",
    "base_forms",
    "variant_map",
}
REQUIRED_NORM_KEYS = {"url", "format", "columns"}
REQUIRED_NORM_COLUMN_KEYS = {
    "raw_score",
    "group",
    "pwa_percentile",
    "control_percentile",
}


def _resource_label(resource: dict[str, Any], source: str | Path | None = None) -> str:
    resource_id = resource.get("resource_id", "<unknown>")
    return f"{resource_id} ({source})" if source else str(resource_id)


def _require_str(resource: dict[str, Any], key: str, source: str | Path | None) -> None:
    value = resource.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Resource {_resource_label(resource, source)} must define non-empty string "
            f"field '{key}'."
        )


def _validate_norms(resource: dict[str, Any], source: str | Path | None = None) -> None:
    norms = resource.get("norms", {})
    label = _resource_label(resource, source)
    if norms in (None, {}):
        return
    if not isinstance(norms, dict):
        raise ValueError(f"Resource {label} field 'norms' must be an object if present.")

    for norm_name, norm_spec in norms.items():
        if not isinstance(norm_name, str) or not norm_name.strip():
            raise ValueError(f"Resource {label} contains a norm with an empty name.")
        if not isinstance(norm_spec, dict):
            raise ValueError(f"Resource {label} norm '{norm_name}' must be an object.")

        missing = REQUIRED_NORM_KEYS - set(norm_spec)
        if missing:
            raise ValueError(
                f"Resource {label} norm '{norm_name}' missing required keys: "
                f"{sorted(missing)}."
            )
        if not isinstance(norm_spec["url"], str) or not norm_spec["url"].strip():
            raise ValueError(f"Resource {label} norm '{norm_name}' requires a non-empty url.")
        if norm_spec["format"] != "csv":
            raise ValueError(
                f"Resource {label} norm '{norm_name}' has unsupported format "
                f"{norm_spec['format']!r}; only 'csv' is currently supported."
            )

        columns = norm_spec["columns"]
        if not isinstance(columns, dict):
            raise ValueError(f"Resource {label} norm '{norm_name}' columns must be an object.")
        missing_columns = REQUIRED_NORM_COLUMN_KEYS - set(columns)
        if missing_columns:
            raise ValueError(
                f"Resource {label} norm '{norm_name}' columns missing required keys: "
                f"{sorted(missing_columns)}."
            )
        for column_key in REQUIRED_NORM_COLUMN_KEYS:
            column_name = columns.get(column_key)
            if not isinstance(column_name, str) or not column_name.strip():
                raise ValueError(
                    f"Resource {label} norm '{norm_name}' columns.{column_key} "
                    "must be a non-empty string."
                )


def validate_resource(resource: dict[str, Any], source: str | Path | None = None) -> None:
    """Validate a target vocabulary resource and raise clear errors on failure."""
    if not isinstance(resource, dict):
        raise ValueError(f"Resource loaded from {source or '<unknown>'} must be an object.")

    missing = REQUIRED_RESOURCE_KEYS - set(resource)
    if missing:
        resource_id = resource.get("resource_id", "<unknown>")
        raise ValueError(
            f"Resource {resource_id} ({source or '<unknown source>'}) missing required "
            f"keys: {sorted(missing)}."
        )

    for key in ("resource_id", "display_name", "language", "task_type"):
        _require_str(resource, key, source)

    label = _resource_label(resource, source)
    base_forms = resource["base_forms"]
    variant_map = resource["variant_map"]

    if not isinstance(base_forms, list):
        raise ValueError(f"Resource {label} field 'base_forms' must be a list.")
    if not base_forms:
        raise ValueError(f"Resource {label} must define at least one base form.")
    if not all(isinstance(base_form, str) and base_form.strip() for base_form in base_forms):
        raise ValueError(f"Resource {label} base_forms must all be non-empty strings.")

    duplicate_base_forms = sorted(
        {base_form for base_form in base_forms if base_forms.count(base_form) > 1}
    )
    if duplicate_base_forms:
        raise ValueError(
            f"Resource {label} has duplicate base_forms: {duplicate_base_forms}."
        )

    if not isinstance(variant_map, dict):
        raise ValueError(f"Resource {label} field 'variant_map' must be an object.")

    base_set = set(base_forms)
    unknown_variant_bases = sorted(set(variant_map) - base_set)
    if unknown_variant_bases:
        raise ValueError(
            f"Resource {label} has variant_map keys not present in base_forms: "
            f"{unknown_variant_bases}."
        )

    variant_to_base: dict[str, str] = {}
    for base_form, variants in variant_map.items():
        if not isinstance(variants, list):
            raise ValueError(
                f"Resource {label} variant_map.{base_form} must be a list of strings."
            )
        duplicate_variants = sorted(
            {variant for variant in variants if variants.count(variant) > 1}
        )
        if duplicate_variants:
            raise ValueError(
                f"Resource {label} variant_map.{base_form} has duplicate variants: "
                f"{duplicate_variants}."
            )
        for variant in variants:
            if not isinstance(variant, str) or not variant.strip():
                raise ValueError(
                    f"Resource {label} variant_map.{base_form} contains a non-string "
                    "or empty variant."
                )
            if variant in base_set and variant != base_form:
                raise ValueError(
                    f"Resource {label} maps variant {variant!r} to base form "
                    f"{base_form!r}, but {variant!r} is also a base form."
                )
            if variant in variant_to_base and variant_to_base[variant] != base_form:
                raise ValueError(
                    f"Resource {label} maps variant {variant!r} to multiple base forms: "
                    f"{variant_to_base[variant]!r} and {base_form!r}."
                )
            variant_to_base[variant] = base_form

    _validate_norms(resource, source)


def _build_reverse_variant_lookup(resource: dict[str, Any]) -> dict[str, str]:
    """Flatten base-form-owned variant lists into a token-to-base-form lookup."""
    reverse_lookup = {str(base_form): str(base_form) for base_form in resource["base_forms"]}
    for base_form, variants in resource.get("variant_map", {}).items():
        for variant in variants:
            reverse_lookup[str(variant)] = str(base_form)
    return reverse_lookup


def _with_runtime_fields(resource: dict[str, Any]) -> dict[str, Any]:
    """Attach derived fields used during matching while keeping JSON human-editable."""
    loaded = dict(resource)
    loaded["_base_form_set"] = set(resource["base_forms"])
    loaded["_reverse_variant_lookup"] = _build_reverse_variant_lookup(resource)
    return loaded


def _json_paths(resource_path: str | Path) -> list[Path]:
    path = Path(resource_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Target vocabulary resource path not found: {path}")
    if path.is_file():
        if path.suffix.lower() != ".json":
            raise ValueError(f"Target vocabulary resource file must be JSON: {path}")
        return [path]
    if path.is_dir():
        paths = sorted(path.glob("*.json"))
        if not paths:
            raise FileNotFoundError(f"No JSON resource files found in: {path}")
        return paths
    raise ValueError(f"Target vocabulary resource path is not a file or directory: {path}")


def load_resources_from_path(resource_path: str | Path) -> dict[str, dict[str, Any]]:
    """Load custom target vocabulary resources from a JSON file or directory."""
    resources: dict[str, dict[str, Any]] = {}
    for path in _json_paths(resource_path):
        with path.open("r", encoding="utf-8") as f:
            resource = json.load(f)
        validate_resource(resource, source=path)
        resource_id = resource["resource_id"]
        if resource_id in resources:
            raise ValueError(
                f"Duplicate target vocabulary resource ID {resource_id!r} in {resource_path}."
            )
        resources[resource_id] = _with_runtime_fields(resource)
    return resources


@lru_cache(maxsize=1)
def load_builtin_resources() -> dict[str, dict[str, Any]]:
    """Load bundled target vocabulary resources keyed by resource ID."""
    resources = load_resources_from_path(BUILTIN_RESOURCE_DIR)
    if not resources:
        raise RuntimeError(f"No built-in target vocabulary resources found in {BUILTIN_RESOURCE_DIR}")
    return resources


@lru_cache(maxsize=16)
def _load_custom_resources_cached(resource_path: str) -> dict[str, dict[str, Any]]:
    return load_resources_from_path(resource_path)


def load_target_vocabulary_resources(
    resource_path: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Load target vocabulary resources.

    Without a path, bundled resources are used. With a path, bundled resources
    remain available and custom resources are merged in. If a custom resource
    reuses a bundled resource ID, the custom definition overrides the bundled one.
    """
    if resource_path is None or str(resource_path).strip() == "":
        return load_builtin_resources()

    builtins = dict(load_builtin_resources())
    custom_resources = _load_custom_resources_cached(str(Path(resource_path).expanduser()))
    overlapping_ids = sorted(set(builtins) & set(custom_resources))

    if overlapping_ids:
        logger_msg = (
            "Custom target vocabulary resources override bundled defaults for resource IDs: "
            f"{overlapping_ids}"
        )
        try:
            from psair.core.logger import logger

            logger.info(logger_msg)
        except Exception:
            pass

    builtins.update(custom_resources)
    return builtins


def get_resource(
    resource_id: str,
    resources: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Return one target vocabulary resource by resource ID, if present."""
    return (resources or load_builtin_resources()).get(resource_id)


def get_resource_ids(resources: dict[str, dict[str, Any]] | None = None) -> set[str]:
    """Return resource IDs for the active target vocabulary resources."""
    return set(resources or load_builtin_resources())


def get_builtin_resource(resource_id: str) -> dict[str, Any] | None:
    """Return one bundled target vocabulary resource by resource ID, if present."""
    return get_resource(resource_id, load_builtin_resources())


def get_builtin_resource_ids() -> set[str]:
    """Return resource IDs for bundled target vocabulary resources."""
    return get_resource_ids(load_builtin_resources())
