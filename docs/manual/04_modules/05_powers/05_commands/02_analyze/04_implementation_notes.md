# `powers analyze` Implementation Notes

`powers analyze` dispatches to `analyze_powers_coding()`.

## Dispatch Path

The main path is:

1. `src/diaad/cli/commands.py` registers `powers analyze`.
2. `src/diaad/cli/dispatch.py` dispatches it as a POWERS command.
3. `src/diaad/core/run_context.py` threads input/output paths, the configured coding filename, blinding config, and sample identifier column.
4. `src/diaad/core/run_wrappers.py` calls `analyze_powers_coding()`.
5. `src/diaad/coding/powers/analysis.py` writes the analysis workbook.

## Input Discovery

The command uses exact filename discovery for the configured POWERS coding workbook. The default is:

```text
powers_coding.xlsx
```

If multiple exact matches are found across the input directory and current run output directory, DIAAD raises an actionable multiple-file error.

## Turn Labels

The analysis path inserts `turn_label` immediately after `turn_type`. Valid turn types are `T`, `MT`, `ST`, and `NV`.

Labels are assigned in workbook row order. Invalid or blank turn types inherit the prior label when possible; otherwise they are marked `X`.

## Summary Levels

The analysis workbook is built from `compute_level_summaries()`:

- `Utterances`: labeled utterance data;
- `Turns`: grouped by sample, speaker, and turn label;
- `Speakers`: grouped by sample and speaker with derived turn and ratio metrics;
- `Dialogs`: grouped by sample.

Section E is not read by this path.

## Relevant Sources

- `src/diaad/coding/powers/analysis.py`
- `tests/test_coding/test_powers/test_identifiers.py`
