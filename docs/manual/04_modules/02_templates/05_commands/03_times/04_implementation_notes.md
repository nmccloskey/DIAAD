# `templates times` Implementation Notes

`templates times` dispatches to `make_speaking_time_template_files()`.

## Dispatch Path

The main path is:

1. `src/diaad/cli/commands.py` registers `templates times`.
2. `src/diaad/cli/dispatch.py` marks it as requiring transcript tables.
3. `src/diaad/core/run_context.py` threads input/output paths, sample identifier column, and transcript table filename.
4. `src/diaad/core/run_wrappers.py` calls `make_speaking_time_template_files()`.
5. `src/diaad/coding/templates/times.py` writes the workbook.

## Data Preparation

The command reads the transcript table `samples` sheet, keeps the configured sample identifier column, drops duplicate sample IDs, and sorts the output by sample identifier.

It then adds a blank `speaking_time` column.

## Output File

The current implementation writes:

```text
coding_templates/speaking_times.xlsx
```

The workbook sheet name is `coding_template`.

Unlike later rate commands, this template command does not currently use `advanced.speaking_time_filename` or `advanced.speaking_time_column` to change the generated filename or column name.

## Downstream Interpretation

Shared rate utilities read speaking time as seconds and derive `speaking_minutes` by dividing by 60. Duplicate sample IDs are collapsed by summing `speaking_time`.

## Relevant Sources

- `src/diaad/coding/templates/times.py`
- `src/diaad/coding/utils/rates.py`
- `tests/test_coding/test_templates/test_identifiers.py`
