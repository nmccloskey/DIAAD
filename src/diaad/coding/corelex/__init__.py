"""Target vocabulary coverage analysis with built-in CoreLex-style resources."""

from importlib import import_module

_CORELEX_EXPORTS = {
    "DETAIL_COLUMNS",
    "SUMMARY_COLUMNS",
    "base_columns",
    "compute_corelex_for_text",
    "compute_target_vocabulary_coverage_for_text",
    "extract_corelex_inputs_from_sample_df",
    "run_corelex",
}

_RESOURCE_EXPORTS = {
    "get_builtin_resource",
    "get_builtin_resource_ids",
    "load_builtin_resources",
}

_UTIL_EXPORTS = {
    "extract_transcript_data",
    "get_percentiles",
    "id_core_words",
    "prepare_corelex_inputs",
    "preload_corelex_norms",
    "reformat",
}

__all__ = sorted(_CORELEX_EXPORTS | _RESOURCE_EXPORTS | _UTIL_EXPORTS)


def __getattr__(name):
    if name in _CORELEX_EXPORTS:
        corelex = import_module("diaad.coding.corelex.corelex")
        return getattr(corelex, name)
    if name in _RESOURCE_EXPORTS:
        resources = import_module("diaad.coding.corelex.resources")
        return getattr(resources, name)
    if name in _UTIL_EXPORTS:
        utils = import_module("diaad.coding.corelex.utils")
        return getattr(utils, name)
    raise AttributeError(f"module 'diaad.coding.corelex' has no attribute {name!r}")
