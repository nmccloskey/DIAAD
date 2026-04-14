"""Target vocabulary coverage analysis with built-in CoreLex-style resources."""

from importlib import import_module

_TARGET_VOCAB_EXPORTS = {
    "DETAIL_COLUMNS",
    "SUMMARY_COLUMNS",
    "base_columns",
    "compute_target_vocab_for_text",
    "compute_target_vocabulary_coverage_for_text",
    "extract_target_vocab_inputs_from_sample_df",
    "run_target_vocab",
}

_RESOURCE_EXPORTS = {
    "get_builtin_resource",
    "get_builtin_resource_ids",
    "get_resource",
    "get_resource_ids",
    "load_builtin_resources",
    "load_resources_from_path",
    "load_target_vocabulary_resources",
    "validate_resource",
}

_UTIL_EXPORTS = {
    "extract_transcript_data",
    "get_percentiles",
    "id_core_words",
    "prepare_target_vocab_inputs",
    "preload_target_vocab_norms",
    "reformat",
}

__all__ = sorted(_TARGET_VOCAB_EXPORTS | _RESOURCE_EXPORTS | _UTIL_EXPORTS)


def __getattr__(name):
    if name in _TARGET_VOCAB_EXPORTS:
        target_vocab = import_module("diaad.coding.target_vocab.analysis")
        return getattr(target_vocab, name)
    if name in _RESOURCE_EXPORTS:
        resources = import_module("diaad.coding.target_vocab.resources")
        return getattr(resources, name)
    if name in _UTIL_EXPORTS:
        utils = import_module("diaad.coding.target_vocab.utils")
        return getattr(utils, name)
    raise AttributeError(f"module 'diaad.coding.target_vocab' has no attribute {name!r}")
