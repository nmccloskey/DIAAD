# `blinding encode` Usage Guide

Use `diaad blinding encode` when a prepared workbook should be masked before coding, review, statistical work, or sharing.

## Workflow Placement

For manual coding workflows, the usual sequence is:

```text
prepare coding file
blinding encode
manual coding on blinded workbook
blinding decode
DIAAD analysis
```

After analysis, projects may run `blinding encode` again on selected exports when downstream statistical workflows should remain blinded.

## File Discovery

The command recursively searches the configured input directory for:

```text
*blind_codebook*.xlsx
```

If `advanced.codebook_filename` is set, DIAAD uses that configured codebook filename instead.

It then selects one non-codebook `.xlsx` target file. The command excludes files whose stem contains `blind_codebook` and temporary Excel files beginning with `~$`.

Use a dedicated input directory for the workbook you intend to encode. If multiple non-codebook `.xlsx` files are present, discovery can fail because DIAAD requires one unambiguous target.

## New Codebook Behavior

If no codebook is found, DIAAD generates a new integer blind codebook for columns listed in:

```text
advanced.blind_columns
```

The default configured blind column is:

```text
sample_id
```

Columns requested in `blind_columns` but absent from the target workbook are skipped with a warning. If none of the requested columns are present, encoding fails.

## Existing Codebook Behavior

If a codebook is found, DIAAD uses the `column` values in that codebook to decide which target columns to encode. The configured `blind_columns` list is not the controlling list in this case.

The codebook must cover all observed values in the target workbook for the encoded columns. Reusing a codebook is recommended when multiple blinded files must share the same blind identifiers.

## Outputs

The blinded workbook drops the raw columns and keeps suffixed columns such as `sample_id_blinded`. The diagnostics workbook keeps join or identifier columns, raw values, and blinded values for review.

The codebook has the core columns:

```text
column
raw_value
blind_code
```

## Common Problems

If the wrong workbook is encoded, move unrelated `.xlsx` files out of the input directory and rerun.

If encoding fails because no requested columns are present, check `advanced.blind_columns` and the workbook header row.

If the same sample receives different blind codes across files, use the same codebook for all related encoding runs.

## Read Next

- `blinding decode` quickstart: `docs/manual/04_modules/08_blinding/05_commands/02_decode/01_quickstart.md`
- Exact file-name matching: `docs/manual/03_features/03_exact_file_name_matching.md`
- Blinding implementation notes: `docs/manual/04_modules/08_blinding/04_implementation_notes.md`
