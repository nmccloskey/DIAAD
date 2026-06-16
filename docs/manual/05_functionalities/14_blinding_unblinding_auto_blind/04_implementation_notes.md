# Blinding, Unblinding, and Auto-Blind Implementation Notes

Blinding uses standalone command wrappers in `src/diaad/blinding/` and shared encode/decode helpers in `src/diaad/metadata/`.

## Source Anchors

Primary sources:

- `src/diaad/blinding/encode.py`
- `src/diaad/blinding/decode.py`
- `src/diaad/metadata/blinding.py`
- `src/diaad/metadata/unblinding.py`
- `src/diaad/core/config.py`
- `src/diaad/core/run_context.py`

Relevant tests:

- `tests/test_blinding/test_commands.py`
- `tests/test_metadata/test_blinding.py`
- `tests/test_metadata/test_unblinding.py`

## Configuration

Important advanced settings include:

```yaml
advanced:
  auto_blind: false
  blind_columns:
    - sample_id
  metadata_source: transcript_tables.xlsx
  id_columns:
    - sample_id
    - utterance_id
  codebook_filename: ''
```

`blind_columns` defines columns to encode. `id_columns` defines record identity for metadata recovery and codebook behavior. `codebook_filename` can force discovery of a specific codebook.

## Standalone Encoding

`encode_blinding()` creates:

```text
blinding/<target_stem>_blinded.xlsx
blinding/<target_stem>_blinding_diagnostics.xlsx
blinding/blind_codebook.xlsx
```

If `codebook_filename` is configured, DIAAD resolves that exact filename. Otherwise it searches for an existing workbook whose stem contains `blind_codebook`. If no codebook is found, it generates a new integer codebook for present configured blind columns.

Standalone analysis-style encoding drops raw blind columns from the blinded output and keeps suffixed columns such as `sample_id_blinded`. The diagnostics workbook preserves raw and blinded values for review.

## Standalone Decoding

`decode_blinding()` finds a blind codebook and one non-codebook `.xlsx` target file. It validates the codebook and writes:

```text
blinding/<target_stem>_decoded.xlsx
```

`unblind_dataframe()` supports two patterns:

- suffixed columns, such as `sample_id_blinded`, decoded to `sample_id`;
- in-place blind codes, such as a `sample_id` column whose values are blind codes.

## Auto-Blind Helpers

Lower-level helpers support two workflow styles:

- analysis blinding, which appends suffixed blind columns and removes raw blind columns from the public output;
- coding-file blinding, which replaces configured columns in place for coder-facing files.

Supported module commands call these helpers through run-context configuration. Command pages document whether a specific workflow applies auto-blinding.

## Boundaries

DIAAD blinding masks configured tabular values. It does not scan free text, inspect external files for identifying content, or guarantee practical coder masking.

## Read Next

- Blinding command implementation notes: `docs/manual/04_modules/08_blinding/05_commands/`
- Configurable identifiers implementation notes: `docs/manual/05_functionalities/10_configurable_sample_utterance_identifiers/04_implementation_notes.md`
- File discovery implementation notes: `docs/manual/05_functionalities/09_configured_filenames_file_discovery_input_selection/04_implementation_notes.md`
