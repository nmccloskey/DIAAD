# Blinding Implementation Notes

The Blinding module is implemented under `src/diaad/blinding/` with shared helpers under `src/diaad/metadata/`.

## Encoding

`blinding encode` finds a target workbook and either reuses an existing codebook or generates a new one for the configured blind columns. It writes:

- a blinded workbook;
- a diagnostics workbook;
- a blind codebook when needed.

The lower-level metadata helper can generate integer blind codes, validate codebook compatibility, and replace configured columns with blinded versions.

## Decoding

`blinding decode` finds a blind codebook and a target workbook, validates that the codebook supports the requested restoration, and writes a decoded workbook.

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
