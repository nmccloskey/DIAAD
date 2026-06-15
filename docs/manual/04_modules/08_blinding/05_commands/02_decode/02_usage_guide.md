# `blinding decode` Usage Guide

Use `diaad blinding decode` when a blinded workbook needs to be restored to original identifiers.

## Workflow Placement

For most DIAAD manual-coding workflows, decoding should happen before analysis. Many analysis commands expect canonical sample identifiers, metadata-compatible IDs, or exact workbook relationships that are easier to manage after decoding.

After analysis, users may encode selected analysis outputs again for blinded statistical workflows. Keep decoded canonical analysis files and the codebook in controlled storage so outputs can be traced and reproduced.

## File Discovery

The command recursively searches the configured input directory for a blind codebook. If `advanced.codebook_filename` is set, DIAAD uses that configured filename instead.

It then selects one non-codebook `.xlsx` target file. Use a dedicated input directory containing only the blinded workbook and its codebook when possible.

## What Can Be Decoded

The decoder supports two common patterns.

Analysis-style blinded columns use the configured suffix:

```text
sample_id_blinded
```

The decoded output restores:

```text
sample_id
```

In-place blinded columns are also supported. For example, a coder-facing workbook may contain a `sample_id` column whose values are blind codes. If the codebook contains mappings for `sample_id`, the decoder restores the raw values in that same column.

## Codebook Requirements

The codebook must have one unambiguous mapping for each `(column, raw_value)` pair and each `(column, blind_code)` pair. Duplicate mappings are rejected.

If a codebook covers columns that are not present in the target workbook, the decoder skips those absent columns in non-strict mode.

## Output

The decoded workbook is written under:

```text
blinding/
```

For a target workbook named `analysis_blinded.xlsx`, the output is:

```text
analysis_blinded_decoded.xlsx
```

## Common Problems

If decoding fails because no codebook is found, confirm that the codebook filename contains `blind_codebook` or set `advanced.codebook_filename`.

If no columns are decoded, check whether the target workbook uses the expected `_blinded` suffix or contains in-place blind codes in columns listed in the codebook.

If restored identifiers are blank, the target workbook may contain blind codes that are not covered by the codebook used for decoding.

## Read Next

- `blinding encode` usage guide: `docs/manual/04_modules/08_blinding/05_commands/01_encode/02_usage_guide.md`
- Blinding implementation notes: `docs/manual/04_modules/08_blinding/04_implementation_notes.md`
- Generated Example I/O: `docs/manual/03_features/04_generated_example_io.md`
