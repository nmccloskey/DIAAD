# `cus evaluate` Implementation Notes

`cus evaluate` dispatches to `evaluate_cu_reliability()`.

## Dispatch Path

The main path is:

1. `src/diaad/cli/commands.py` registers `cus evaluate`.
2. `src/diaad/cli/dispatch.py` dispatches it as a CU command.
3. `src/diaad/core/run_context.py` threads input/output paths, `cu_paradigms`, and identifier columns.
4. `src/diaad/core/run_wrappers.py` calls `evaluate_cu_reliability()`.
5. `src/diaad/coding/compl_utts/rel_evaluation.py` writes reliability outputs.

## Input Discovery

The command uses exact filename discovery for:

```text
cu_coding.xlsx
cu_reliability_coding.xlsx
```

It searches the input directory and current run output directory.

## Column Resolution

For each paradigm, the evaluator resolves either an unprefixed primary-vs-reliability schema or a `c2`/`c3` schema. Merged output columns are canonicalized to `c2_*` and `c3_*` so downstream summarization does not depend on the source schema.

Rows are merged by the configured utterance identifier. Coverage diagnostics compare represented rows against the primary coding file.

## Metrics

The evaluator computes:

- raw agreement and Cohen's kappa on utterance-level categorical ratings;
- ICC(2,1) and ICC(2,k) on sample-level totals;
- variance diagnostics for paired ratings and totals;
- legacy sample-level agreement summaries.

Kappa and ICC can be `NaN` when data are insufficient or variance is too limited for the metric to be informative.

## Relevant Sources

- `src/diaad/coding/compl_utts/rel_evaluation.py`
- `src/diaad/coding/compl_utts/analysis.py`
- `src/diaad/coding/utils/rel_eval_utils.py`
- `tests/test_coding/test_compl_utts/test_identifiers.py`
