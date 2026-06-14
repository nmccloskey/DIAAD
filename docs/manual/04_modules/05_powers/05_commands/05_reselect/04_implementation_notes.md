# `powers reselect` Implementation Notes

`powers reselect` dispatches to `reselect_powers_rel()`.

## Dispatch Path

The main path is:

1. `src/diaad/cli/commands.py` registers `powers reselect`.
2. `src/diaad/cli/dispatch.py` dispatches it as a POWERS command.
3. `src/diaad/core/run_context.py` threads metadata fields, reliability fraction, random seed, automation settings, configured filenames, and sample identifier column.
4. `src/diaad/core/run_wrappers.py` calls `reselect_powers_rel()`.
5. `src/diaad/coding/powers/rel_reselection.py` writes the replacement workbook.

## Pair Discovery

The command uses the shared reliability-pair discovery utility with the configured primary and reliability filename globs. Defaults are:

```text
*powers_coding.xlsx
*powers_reliability_coding.xlsx
```

Configured metadata fields are used to pair original and reliability files when available.

## Reselection Logic

The command collects sample IDs already present in reliability files, selects unused sample IDs from the original workbook, and writes selected rows to:

```text
reselected_powers_reliability/
```

The default output filename for the default primary workbook is:

```text
reselected_powers_reliability_coding.xlsx
```

The builder clears manual POWERS fields, ensures administrative reliability columns exist, and can rerun POWERS automation before writing.

## Relevant Sources

- `src/diaad/coding/powers/rel_reselection.py`
- `src/diaad/coding/powers/automation.py`
- `src/diaad/coding/utils/reselection_utils.py`
- `tests/test_coding/test_powers/test_identifiers.py`
