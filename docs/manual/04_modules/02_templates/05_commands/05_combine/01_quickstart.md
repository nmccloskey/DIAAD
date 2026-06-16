# `templates combine` Quickstart

`diaad templates combine` stacks multiple same-schema Excel template workbooks into one workbook. It is useful when coding templates were completed in separate files and need to be brought back together for review or downstream handling.

## Run

```bash
diaad templates combine --config config
```

## Minimum Inputs

Point `project.input_dir` at a folder containing one or more `.xlsx` workbooks. DIAAD searches that folder recursively and ignores temporary Excel lock files beginning with `~$`.

All discovered workbooks must have the same sheet names. Within each matching sheet, all workbooks must have the same column names.

## Primary Output

By default, the command writes:

```text
coding_templates/
  combined.xlsx
```

The output workbook mirrors the input sheet names and adds:

| Column | Purpose |
|---|---|
| `combined_id` | A new row identifier, numbered `1..n` within each combined sheet. |
| `source_file` | The input workbook path relative to `input_dir`. |

It also adds a `metadata` sheet with one row per input workbook and sheet:

```text
source_file
order
sheet
num_rows
```

## Immediate Next Step

Inspect `metadata` first to confirm every expected workbook contributed rows to each sheet, then inspect the combined data sheets for coder-specific or site-specific review.

## Read Next

- Templates module quickstart: `docs/manual/04_modules/02_templates/01_quickstart.md`
- Exact file name matching: `docs/manual/03_features/03_exact_file_name_matching.md`
- Configuration: `docs/manual/02_operation/04_configuration.md`
