# `templates combine` Implementation Notes

`templates combine` dispatches to `make_combined_template_file()`.

## Dispatch Path

The main path is:

1. `src/diaad/cli/commands.py` registers `templates combine`.
2. `src/diaad/cli/dispatch.py` dispatches it without requiring transcript tables.
3. `src/diaad/core/run_context.py` threads standard input/output paths.
4. `src/diaad/core/run_wrappers.py` calls `make_combined_template_file()`.
5. `src/diaad/coding/templates/combination.py` reads same-schema workbooks and writes `combined.xlsx`.

## Input Discovery

The implementation uses `find_files_by_extension()` from `src/diaad/metadata/discovery.py` with `.xlsx`. That DIAAD helper delegates to PSAIR's `find_matching_files()` backend while preserving duplicate filenames and Excel lock-file skipping.

Discovery is recursive under the configured `input_dir`, preserves duplicate filenames as distinct paths, and skips temporary Excel files beginning with `~$`. The configured `output_dir` is not added as a second input search root.

## Validation

The first discovered workbook establishes the expected sheet names and per-sheet columns.

Each later workbook must have the same sheet-name set. Each corresponding sheet must have the same column-name set. If a workbook differs, DIAAD raises a validation error rather than producing a partial combined file.

The input sheet name `metadata` is reserved for output provenance. The input column names `combined_id` and `source_file` are reserved for generated output columns.

## Output

The command writes:

```text
coding_templates/combined.xlsx
```

Each user data sheet is written with:

```text
combined_id
source_file
<original input columns>
```

`combined_id` is generated separately for each output sheet. `source_file` is the path relative to `input_dir`, which lets duplicate filenames remain distinguishable.

The `metadata` sheet is built from the same read pass and records `source_file`, `order`, `sheet`, and `num_rows`.

## Relevant Sources

- `src/diaad/coding/templates/combination.py`
- `src/diaad/metadata/discovery.py`
- `tests/test_coding/test_templates/test_combination.py`
- `tests/test_metadata/test_discovery.py`
