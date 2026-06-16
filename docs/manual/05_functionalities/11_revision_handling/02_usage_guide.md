# Revision Handling Usage Guide

Revision handling is the practice of deciding what must be rerun, recoded, or rechecked after transcript tables or coding files change.

## Low-Risk Revisions

Some edits may be low risk when they do not change analytic units:

- correcting a typo in a metadata value before downstream files are generated;
- fixing an output filename or folder placement before the file is used;
- adding a note column that downstream commands ignore.

Even low-risk revisions should be documented if they occur after a run has produced analysis outputs.

## Higher-Risk Revisions

These revisions usually require downstream review:

- changing `sample_id` or `utterance_id` values;
- changing utterance text after manual coding;
- adding, deleting, or splitting utterance rows;
- changing speaker labels after speaker exclusion has been used;
- changing metadata fields used for grouping, blinding, or resource matching;
- changing file names that DIAAD discovers by exact configured filename;
- editing completed coding files after reliability evaluation.

The safest response is often to regenerate affected derived files from the revised table and redo any manual coding that depended on changed rows.

## Transcript Table Edits

When editing transcript tables:

- keep `sample_id` stable when the sample is the same;
- keep `utterance_id` stable when the utterance row is the same analytic unit;
- use `position` and `position_sub` to preserve utterance order;
- avoid changing identifier column names after downstream files exist;
- review `metadata_mismatch` before treating metadata as complete.

If you insert or delete utterance rows after coding files exist, review all utterance-level coding, reliability, and analysis files that may no longer align with the transcript table.

## Exporting Revised CHAT Files

Use `diaad transcripts chats` only when you need CHAT-style files derived from the current transcript table. The exported files are useful for revised-transcript circulation, but they do not replace the need to manage downstream DIAAD artifacts.

## Practical Recovery Pattern

When uncertain, use this conservative path:

1. Archive the earlier output directory.
2. Revise the transcript table.
3. Run a new DIAAD command from the revised input.
4. Compare generated coding files with previous coding files.
5. Recode or reconcile affected rows.
6. Rerun reliability and analysis.
7. Keep the new timestamped output as the current analysis state.

## Read Next

- Run provenance and audit artifacts: `docs/manual/05_functionalities/03_run_provenance_audit_artifacts/02_usage_guide.md`
- Reliability selection and evaluation: `docs/manual/05_functionalities/12_reliability_selection_evaluation_reselection/02_usage_guide.md`
- Blinding functionality: `docs/manual/05_functionalities/14_blinding_unblinding_auto_blind/02_usage_guide.md`
