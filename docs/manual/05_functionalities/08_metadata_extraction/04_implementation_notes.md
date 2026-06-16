# Metadata Extraction Implementation Notes

Metadata extraction is built from DIAAD configuration normalization, PSAIR metadata matching, transcript-table construction, and metadata-loading helpers used by later workflows.

## Source Anchors

Primary sources:

- `src/diaad/core/config.py`
- `src/diaad/core/run_context.py`
- `src/diaad/transcripts/transcript_tables.py`
- `src/diaad/metadata/utils.py`
- `src/diaad/metadata/discovery.py`

Relevant tests:

- `tests/test_transcripts/test_transcript_tables.py`
- `tests/test_metadata/test_utils.py`
- `tests/test_metadata/test_blinding.py`
- `tests/test_metadata/test_unblinding.py`

## Configuration Parsing

`project.metadata_fields` must be a dictionary. Field definitions may be:

- a non-empty string, interpreted as a metadata pattern;
- a list of strings, interpreted as allowed or searchable values.

Invalid metadata field types raise configuration errors before processing.

## Run Context Setup

`RunContext._build_metadata_field_state()` builds a metadata manager from the configured metadata fields and the resolved input directory. The resulting metadata field objects are passed to transcript tabularization.

## Table Construction

During `tabularize_transcripts()`, each metadata field is evaluated for each CHAT file path. Field names become columns in the `samples` sheet.

The reserved column name `metadata_mismatch` cannot be used as a metadata field name because DIAAD uses it for diagnostics.

If a metadata field does not match, DIAAD:

- writes a blank value for that field;
- sets `metadata_mismatch` to `1` for the sample;
- writes a row in `metadata_mismatches` with the sample identifier, file context, metadata field, source path, reason, and written value.

## Loading Metadata Later

`load_metadata_from_transcript_tables()` can load joined transcript metadata from explicit transcript table paths or discover a table by configured filename. It calls `extract_transcript_data(kind="joined")`, which merges `samples` and `utterances` on the configured sample identifier field.

When multiple explicit transcript tables are provided and `combine=True`, the helper concatenates their joined data. When discovery is used without explicit paths, DIAAD still applies its strict configured-filename discovery policy.

## Boundary With PSAIR

DIAAD delegates low-level metadata matching to PSAIR metadata helpers. DIAAD documentation should focus on the configured fields, visible transcript-table columns, mismatch diagnostics, and downstream effects.

## Read Next

- Configuration implementation notes: `docs/manual/05_functionalities/01_configuration_sources_defaults_overrides/04_implementation_notes.md`
- Transcript preprocessing implementation notes: `docs/manual/05_functionalities/06_transcript_preprocessing_tabularization_chat_export/04_implementation_notes.md`
- File discovery implementation notes: `docs/manual/05_functionalities/09_configured_filenames_file_discovery_input_selection/04_implementation_notes.md`
