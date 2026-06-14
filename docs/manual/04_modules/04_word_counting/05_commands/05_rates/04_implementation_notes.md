# `words rates` Implementation Notes

`words rates` dispatches to `calculate_word_count_rates()`.

## Dispatch Path

The main path is:

1. `src/diaad/cli/commands.py` registers `words rates`.
2. `src/diaad/cli/dispatch.py` dispatches it as a word-counting command.
3. `src/diaad/core/run_context.py` threads input/output paths, sample-summary filename, speaking-time filename and column, and sample identifier column.
4. `src/diaad/core/run_wrappers.py` calls `calculate_word_count_rates()`.
5. `src/diaad/coding/word_counts/rates.py` writes rate outputs.

## Input Discovery

The command reads the configured word-count sample summary. The default filename is:

```text
word_counting_by_sample.xlsx
```

It also reads the configured speaking-time workbook. The default filename is:

```text
speaking_times.xlsx
```

## Merge And Rate Calculation

The speaking-time reader interprets the configured speaking-time column as seconds and creates `speaking_minutes`.

Duplicate speaking-time rows for the same sample are summed before merging. The final output computes:

```text
total_words_per_min = total_words / speaking_minutes
```

The result is rounded to three decimal places.

## Relevant Sources

- `src/diaad/coding/word_counts/rates.py`
- `src/diaad/coding/utils/rates.py`
- `tests/test_coding/test_word_counts/test_identifiers.py`
