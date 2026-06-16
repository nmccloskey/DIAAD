# Transcription-Based Workflow Baseline Implementation Notes

The baseline workflow is a sequence across transcript selection, reliability evaluation, and transcript table construction. It is not implemented as one omnibus DIAAD command.

## Source Anchors

Primary sources:

- `src/diaad/transcripts/cha_files.py`
- `src/diaad/transcripts/transcript_tables.py`
- `src/diaad/transcripts/transcription_reliability_selection.py`
- `src/diaad/transcripts/transcription_reliability_evaluation.py`
- `src/diaad/transcripts/detabularization.py`
- `src/diaad/core/run_context.py`
- `src/diaad/metadata/discovery.py`

Relevant tests:

- `tests/test_transcripts/test_cha_files.py`
- `tests/test_transcripts/test_transcript_tables.py`
- `tests/test_transcripts/test_detabularization.py`

## Command Sequence

The baseline uses these command families:

```text
transcripts select
transcripts evaluate
transcripts reselect
transcripts tabularize
transcripts chats
```

`transcripts chats` is part of the broader transcript-table lifecycle, but it is usually needed only after table revision or when exported CHAT files are required.

## Transcript Table Output

`transcripts tabularize` writes a workbook with:

```text
samples
utterances
metadata_mismatches
```

The `utterances` sheet includes `position` and `position_sub`, which downstream export uses to reconstruct utterance order.

## Prerequisite Behavior

Many transcript-derived commands call transcript-table prerequisite logic before dispatch. If `transcripts tabularize` is not part of the same run, DIAAD searches the configured input directory and current run output directory for `advanced.transcript_table_filename`.

`project.auto_tabularize` can allow later commands to create transcript tables when missing, but the recommended research workflow is explicit: tabularize, inspect, and then run downstream commands against the reviewed table.

## Boundary

The workflow creates and checks a transcript scaffold. It does not decide whether a transcription convention is valid, whether a reliability threshold is sufficient, or whether later coding manuals are appropriate.

## Read Next

- `transcripts tabularize` implementation notes: `docs/manual/04_modules/01_transcripts/05_commands/01_tabularize/04_implementation_notes.md`
- `transcripts evaluate` implementation notes: `docs/manual/04_modules/01_transcripts/05_commands/04_evaluate/04_implementation_notes.md`
- Configuration: `docs/manual/02_operation/04_configuration.md`
