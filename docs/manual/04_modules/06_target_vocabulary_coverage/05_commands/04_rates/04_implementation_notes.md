# `vocab rates` Implementation Notes

`vocab rates` dispatches to `calculate_target_vocab_rates()`.

## Dispatch Path

The main path is:

1. `src/diaad/cli/commands.py` registers `vocab rates`.
2. `src/diaad/cli/dispatch.py` dispatches it as a target-vocabulary command.
3. `src/diaad/core/run_context.py` threads input/output paths and sample identifier column.
4. `src/diaad/core/run_wrappers.py` calls `calculate_target_vocab_rates()`.
5. `src/diaad/coding/target_vocab/rates.py` writes the rates workbook.

## Analysis Discovery

The command recursively locates analysis workbooks matching:

```text
target_vocab_data_*.xlsx
```

It reads each workbook's `summary` sheet and adds `source_file` before concatenating rows.

## Rate Calculation

The command coerces `speaking_time` to numeric seconds and creates:

```text
speaking_minutes
```

It infers numeric count-like numerator columns, adds `_per_min` columns, rounds rates to three decimal places, and writes:

```text
target_vocab/
  target_vocab_rates.xlsx
```

## Relevant Sources

- `src/diaad/coding/target_vocab/rates.py`
- `src/diaad/coding/utils/rates.py`
- `tests/test_coding/test_target_vocab/test_identifiers.py`
