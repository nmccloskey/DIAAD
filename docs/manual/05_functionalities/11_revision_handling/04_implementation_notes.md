# Revision Handling Implementation Notes

DIAAD supports revision handling through stable identifiers, timestamped outputs, transcript-table detabularization, and strict file discovery. It does not currently provide a full dependency tracker that automatically invalidates downstream manual coding files after a transcript table changes.

## Source Anchors

Primary sources:

- `src/diaad/core/run_context.py`
- `src/diaad/core/provenance.py`
- `src/diaad/transcripts/transcript_tables.py`
- `src/diaad/transcripts/detabularization.py`
- module-specific coding file, reliability, and analysis modules under `src/diaad/coding/`

Relevant tests:

- `tests/test_transcripts/test_transcript_tables.py`
- `tests/test_transcripts/test_detabularization.py`
- module-specific reliability and analysis tests

## What DIAAD Does

DIAAD helps revisions stay auditable by:

- writing outputs into timestamped run directories;
- writing effective configuration and run metadata for normal CLI runs;
- preserving transcript-table identifiers when users keep them stable;
- requiring exact configured filenames for many downstream inputs;
- exporting CHAT-style files from revised transcript tables when requested.

## What DIAAD Does Not Do

DIAAD does not automatically:

- compare a revised transcript table to every prior coding file;
- mark coding rows as stale after transcript text changes;
- recalculate whether human coding remains valid;
- migrate manual codes across inserted or deleted utterance rows;
- decide whether a reliability subset must be recoded.

Those decisions remain project-level review tasks.

## Interaction With Generated Files

Commands that generate coding files usually read transcript tables or configured coding workbooks at run time. If the input table has changed, newly generated files reflect the new table. Existing completed coding files remain whatever they were when saved.

This means that a later analysis can be internally consistent but still methodologically stale if it reads an old coding file after the transcript table has changed.

## Detabularization

`detabularize_transcripts()` can export revised transcript tables as CHAT-style files. It sorts by `position` and `position_sub`, writes derived filenames, and records those names in an updated transcript table copy.

This supports revision circulation, but it is not an automated reconciliation process for downstream coding artifacts.

## Review Note

TODO: Before publication, confirm the preferred user-facing procedure for inserting utterances with `position` and `position_sub`, especially when a project needs to preserve already-created utterance identifiers.

## Read Next

- Transcript preprocessing implementation notes: `docs/manual/05_functionalities/06_transcript_preprocessing_tabularization_chat_export/04_implementation_notes.md`
- Run provenance implementation notes: `docs/manual/05_functionalities/03_run_provenance_audit_artifacts/04_implementation_notes.md`
- Testing: `docs/manual/02_operation/05_testing.md`
