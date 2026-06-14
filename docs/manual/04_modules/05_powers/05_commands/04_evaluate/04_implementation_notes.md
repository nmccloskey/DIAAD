# `powers evaluate` Implementation Notes

`powers evaluate` dispatches to `evaluate_powers_reliability()`.

## Dispatch Path

The main path is:

1. `src/diaad/cli/commands.py` registers `powers evaluate`.
2. `src/diaad/cli/dispatch.py` dispatches it as a POWERS command.
3. `src/diaad/core/run_context.py` threads input/output paths, configured filenames, and identifier columns.
4. `src/diaad/core/run_wrappers.py` calls `evaluate_powers_reliability()`.
5. `src/diaad/coding/powers/rel_evaluation.py` writes reliability outputs.

## Input Discovery

The command uses exact filename discovery for the configured primary and reliability workbooks. Defaults are:

```text
powers_coding.xlsx
powers_reliability_coding.xlsx
```

## Merge And Metrics

Primary and reliability rows are merged on the configured sample and utterance identifier columns. The evaluator inserts a synthetic `reliability_id` so ICC targets remain unique even when utterance IDs repeat across samples.

Continuous fields are coerced to numeric values, then row-level absolute difference, percent difference, and percent similarity columns are added. Continuous summaries include ICC and distribution diagnostics.

Categorical summaries are computed for `turn_type` and `collab_repair`. `collab_repair` is reduced to presence or absence before agreement and kappa are calculated.

## Relevant Sources

- `src/diaad/coding/powers/rel_evaluation.py`
- `src/diaad/coding/utils/rel_eval_utils.py`
- `tests/test_coding/test_powers/test_identifiers.py`
