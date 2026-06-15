# Blinding Implementation Notes

The Blinding module is implemented under `src/diaad/blinding/` with shared helpers under `src/diaad/metadata/`.

## Encoding

`blinding encode` finds one target workbook and either reuses an existing codebook or generates a new one for the configured blind columns. Standalone encoding creates analysis-style blinded columns with the configured suffix, currently `_blinded`, and removes the raw columns from the blinded output. It writes:

- a blinded workbook;
- a diagnostics workbook;
- a blind codebook.

The lower-level metadata helper can generate integer blind codes, validate codebook compatibility, append suffixed blinded columns for analysis-style outputs, and replace configured columns in-place for coder-facing files generated through auto-blind workflows.

## Decoding

`blinding decode` finds a blind codebook and one target workbook, validates that the codebook supports the requested restoration, and writes a decoded workbook. It can restore both analysis-style suffixed columns, such as `sample_id_blinded`, and in-place blinded columns, such as a `sample_id` column containing blind codes.

Shared unblinding helpers can also be used inside analysis workflows when `auto_blind` resources are present and outputs need canonical identifiers.

## Configuration

Important advanced settings include:

- `auto_blind`
- `blind_columns`
- `id_columns`
- `metadata_source`
- `codebook_filename`

## Boundaries

The implementation masks configured columns. It does not inspect transcript text for identifying content and does not guarantee privacy or coder masking.
