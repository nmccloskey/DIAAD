# `words evaluate` Implementation Notes

`words evaluate` dispatches to `evaluate_word_count_reliability()`.

## Dispatch Path

The main path is:

1. `src/diaad/cli/commands.py` registers `words evaluate`.
2. `src/diaad/cli/dispatch.py` dispatches it as a word-counting command.
3. `src/diaad/core/run_context.py` threads input/output paths and identifier columns.
4. `src/diaad/core/run_wrappers.py` calls `evaluate_word_count_reliability()`.
5. `src/diaad/coding/word_counts/rel_evaluation.py` writes reliability outputs.

## Input Discovery

The command uses exact filename discovery for:

```text
word_counting.xlsx
word_count_reliability.xlsx
```

It searches the input directory and current run output directory.

## Merge And Filtering

Primary and reliability rows are merged on the configured sample and utterance identifier columns.

The evaluator coerces both word-count columns to numeric values and drops paired rows where either side is missing after coercion.

## Metrics

For each paired utterance, the evaluator computes:

- absolute difference;
- percent difference;
- percent similarity;
- agreement flag;
- ICC(2,1) across paired utterance counts;
- coverage and variance diagnostics for the text report.

The agreement flag is `1` when absolute difference is at most one word or percent similarity is at least 85 percent.

## Relevant Sources

- `src/diaad/coding/word_counts/rel_evaluation.py`
- `src/diaad/coding/utils/rel_eval_utils.py`
- `tests/test_coding/test_word_counts/test_identifiers.py`
