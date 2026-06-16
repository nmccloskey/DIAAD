# Configurable Sample and Utterance Identifiers Implementation Notes

Identifier column names are normalized in configuration and then passed through run-context keyword builders to transcript, coding, reliability, blinding, and analysis functions.

## Source Anchors

Primary sources:

- `src/diaad/core/config.py`
- `src/diaad/core/run_context.py`
- `src/diaad/transcripts/transcript_tables.py`
- `src/diaad/transcripts/detabularization.py`
- `src/diaad/coding/templates/subset.py`
- module-specific coding, reliability, and analysis modules under `src/diaad/coding/`
- `src/diaad/metadata/blinding.py`
- `src/diaad/metadata/unblinding.py`

Relevant tests:

- `tests/test_transcripts/test_transcript_tables.py`
- `tests/test_transcripts/test_detabularization.py`
- `tests/test_coding/test_templates/test_identifiers.py`
- `tests/test_coding/test_convo_turns/test_identifiers.py`
- `tests/test_coding/test_target_vocab/test_identifiers.py`
- `tests/test_metadata/test_blinding.py`

## Configuration Fields

The relevant advanced settings are:

```yaml
advanced:
  sample_id_column: sample_id
  utterance_id_column: utterance_id
```

Configuration validation requires both values to be non-empty strings.

The compatibility properties `sample_id_field` and `utterance_id_field` expose these values to run-context and module code.

## Transcript Tables

`tabularize_transcripts()` accepts `sample_id_field` and `utterance_id_field`. These names become the identifier columns in the `samples` and `utterances` sheets.

`extract_transcript_data()` merges `samples` and `utterances` on the configured sample identifier field for joined transcript data. It raises an error if the required join column is missing.

`detabularize_transcripts()` accepts `sample_id_field` and requires that field in both the `samples` and `utterances` sheets.

## Module Propagation

`RunContext` passes configured identifier names into many module functions through command-specific `kwargs_*` methods. Module implementations generally use the configured sample identifier for sample-level grouping and the configured sample-plus-utterance identifiers for utterance-level joins.

Some legacy or compatibility paths may accept older column names in specific contexts, but user-facing project files should follow the active configuration.

## Blinding Identifier Settings

Blinding also uses:

```yaml
advanced:
  id_columns:
    - sample_id
    - utterance_id
```

These define record identity for codebook and encode/decode behavior. They should be reviewed whenever sample or utterance identifier names are customized.

## Read Next

- Configuration implementation notes: `docs/manual/05_functionalities/01_configuration_sources_defaults_overrides/04_implementation_notes.md`
- Transcript preprocessing implementation notes: `docs/manual/05_functionalities/06_transcript_preprocessing_tabularization_chat_export/04_implementation_notes.md`
- Blinding implementation notes: `docs/manual/05_functionalities/14_blinding_unblinding_auto_blind/04_implementation_notes.md`
