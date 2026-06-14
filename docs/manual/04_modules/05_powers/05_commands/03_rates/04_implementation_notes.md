# `powers rates` Implementation Notes

`powers rates` dispatches to `calculate_powers_rates()`.

## Dispatch Path

The main path is:

1. `src/diaad/cli/commands.py` registers `powers rates`.
2. `src/diaad/cli/dispatch.py` dispatches it as a POWERS command.
3. `src/diaad/core/run_context.py` threads input/output paths, speaking-time filename and column, and sample identifier column.
4. `src/diaad/core/run_wrappers.py` calls `calculate_powers_rates()`.
5. `src/diaad/coding/powers/rates.py` writes the rates workbook.

## Analysis Discovery

The command recursively locates analysis workbooks matching:

```text
*powers*analysis*.xlsx
```

It reads the `Dialogs` sheet from each match, adds `source_file`, and concatenates the rows.

## Rate Calculation

The speaking-time reader interprets the configured speaking-time column as seconds and creates `speaking_minutes`.

Numerator columns are inferred from numeric dialog-summary columns after excluding identifiers, existing speaking-time fields, proportions, and ratios. The command then adds columns with the `_per_min` suffix and rounds values to three decimal places.

## Relevant Sources

- `src/diaad/coding/powers/rates.py`
- `src/diaad/coding/utils/rates.py`
- `tests/test_coding/test_powers/test_identifiers.py`
