# `turns evaluate` Implementation Notes

`turns evaluate` dispatches to `run_evaluate_digital_convo_turns()`.

## Dispatch Path

The main path is:

1. `src/diaad/cli/commands.py` registers `turns evaluate`.
2. `src/diaad/cli/dispatch.py` dispatches the command without a transcript-table prerequisite.
3. `src/diaad/core/run_context.py` passes metadata-field I/O and the configured sample identifier.
4. `src/diaad/core/run_wrappers.py` calls `evaluate_digital_convo_turns_reliability()`.
5. `src/diaad/coding/convo_turns/rel_evaluation.py` writes reliability results, report, and alignments.

## File Discovery

The current implementation uses exact filename discovery for:

```text
conversation_turns.xlsx
conversation_turns_reliability.xlsx
```

It searches the configured input directory and the current run output directory. Metadata fields are passed through the run context, but the current DCT reliability evaluator does not use them for pairing; the exact filenames define the primary and reliability pair.

## Normalization

The evaluator normalizes each workbook to sample/session/bin/turns fields. If `session` or `bin` is missing, it is inserted as a blank field. If duplicate sample/session/bin rows are present, only the first row is kept.

Rows are merged with an outer join, so missing primary or reliability rows are retained for coverage and sequence reporting.

## Metrics

Count agreement is built by expanding each merged row into one row per observed speaker digit, then comparing primary and reliability counts. ICC(2,1) is calculated from those count targets through shared reliability utilities.

Sequence agreement uses the Levenshtein and global-alignment helpers also used by transcription reliability code.

## Relevant Sources

- `src/diaad/coding/convo_turns/rel_evaluation.py`
- `src/diaad/coding/convo_turns/analysis.py`
- `src/diaad/coding/utils/rel_eval_utils.py`
- `src/diaad/transcripts/transcription_reliability_evaluation.py`
- `tests/test_coding/test_convo_turns/test_identifiers.py`
