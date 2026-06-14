# `words reselect` Implementation Notes

`words reselect` dispatches to `reselect_wc_rel()`.

## Dispatch Path

The main path is:

1. `src/diaad/cli/commands.py` registers `words reselect`.
2. `src/diaad/cli/dispatch.py` dispatches it as a word-counting command.
3. `src/diaad/core/run_context.py` threads metadata fields, reliability fraction, random seed, paths, and sample identifier column.
4. `src/diaad/core/run_wrappers.py` calls `reselect_wc_rel()`.
5. `src/diaad/coding/word_counts/rel_reselection.py` writes the replacement workbook.

## Pair Discovery

The command uses the shared reliability-pair discovery utility with:

```text
*word_counting.xlsx
*word_count_reliability.xlsx
```

Configured metadata fields are used to pair original and reliability files when available.

## Reselection Logic

The command collects sample IDs already present in reliability files, selects unused sample IDs from the original workbook, and writes selected rows to:

```text
reselected_word_count_reliability/
  reselected_word_count_reliability.xlsx
```

The output frame preserves the original leading columns, keeps useful reliability-template tail columns, ensures a word-count reliability comment column, and recomputes first-pass counts for selected utterances.

## Relevant Sources

- `src/diaad/coding/word_counts/rel_reselection.py`
- `src/diaad/coding/word_counts/files.py`
- `src/diaad/coding/utils/reselection.py`
