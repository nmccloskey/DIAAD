"""Built-in resources for target vocabulary coverage analysis."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


RESOURCE_DIR = Path(__file__).with_name("resources")
REQUIRED_RESOURCE_KEYS = {
    "id",
    "display_name",
    "language",
    "task_type",
    "base_forms",
    "variant_map",
}


def _validate_resource_minimally(resource: dict[str, Any]) -> None:
    """
    Run first-pass checks for built-in target vocabulary resources.

    This is intentionally light. Future validation can expand this hook into a
    stricter schema check without changing the loader's public shape.
    """
    missing = REQUIRED_RESOURCE_KEYS - set(resource)
    if missing:
        resource_id = resource.get("id", "<unknown>")
        raise ValueError(f"Resource {resource_id} missing required keys: {sorted(missing)}")

    base_forms = resource["base_forms"]
    variant_map = resource["variant_map"]
    if not isinstance(base_forms, list):
        raise ValueError(f"Resource {resource['id']} has non-list base_forms.")
    if not isinstance(variant_map, dict):
        raise ValueError(f"Resource {resource['id']} has non-dict variant_map.")

    base_set = set(base_forms)
    unknown_variant_bases = sorted(set(variant_map) - base_set)
    if unknown_variant_bases:
        raise ValueError(
            f"Resource {resource['id']} has variant_map keys not present in base_forms: "
            f"{unknown_variant_bases}"
        )


def _build_reverse_variant_lookup(resource: dict[str, Any]) -> dict[str, str]:
    """Flatten base-form-owned variant lists into a token -> base_form lookup."""
    reverse_lookup = {str(base_form): str(base_form) for base_form in resource["base_forms"]}
    for base_form, variants in resource.get("variant_map", {}).items():
        for variant in variants:
            reverse_lookup[str(variant)] = str(base_form)
    return reverse_lookup


def _with_runtime_fields(resource: dict[str, Any]) -> dict[str, Any]:
    """Attach derived fields used during matching while keeping JSON simple."""
    loaded = dict(resource)
    loaded["_base_form_set"] = set(resource["base_forms"])
    loaded["_reverse_variant_lookup"] = _build_reverse_variant_lookup(resource)
    return loaded


@lru_cache(maxsize=1)
def load_builtin_resources() -> dict[str, dict[str, Any]]:
    """
    Load bundled target vocabulary coverage resources keyed by resource id.

    Each JSON file represents one built-in CoreLex-style task resource.
    """
    resources: dict[str, dict[str, Any]] = {}
    for path in sorted(RESOURCE_DIR.glob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            resource = json.load(f)
        _validate_resource_minimally(resource)
        resource_id = resource["id"]
        if resource_id in resources:
            raise ValueError(f"Duplicate target vocabulary resource id: {resource_id}")
        resources[resource_id] = _with_runtime_fields(resource)
    return resources


def get_builtin_resource(resource_id: str) -> dict[str, Any] | None:
    """Return one built-in target vocabulary resource by id, if present."""
    return load_builtin_resources().get(resource_id)


def get_builtin_resource_ids() -> set[str]:
    """Return ids for bundled target vocabulary resources."""
    return set(load_builtin_resources())
