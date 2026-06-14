# Transcripts Implementation Notes

The Transcripts module is implemented under `src/diaad/transcripts/` and is dispatched through the normal command registry and run context.

## Runtime Preparation

`transcripts tabularize` and `transcripts select` require CHAT files. Other commands that require transcript tables call the run context's transcript-table prerequisite logic unless tabularization is already part of the same command run.

If a required transcript table is missing and `project.auto_tabularize` is `false`, DIAAD stops with an error. If `auto_tabularize` is `true`, DIAAD loads CHAT files and creates transcript tables automatically in the current run output directory.

## Table Construction

`transcript_tables.py` writes `transcript_tables/transcript_tables.xlsx` by default. The workbook includes:

- `samples`
- `utterances`
- `metadata_mismatches`

Sample IDs and utterance IDs use configurable column names from `advanced.sample_id_column` and `advanced.utterance_id_column`. The default names are `sample_id` and `utterance_id`.

CHAT files are discovered recursively from the input directory. Reliability directories can be excluded where appropriate. Source file paths are stored in a portable input-relative form so later tables can preserve source context.

## CHAT Export

`detabularization.py` finds one configured transcript table, loads `samples` and `utterances`, sorts utterances by `position` and `position_sub`, and writes CHAT-style files under `chat_files/`. It can use a `*template_header.cha` file if one is provided; otherwise it uses a default CHAT header.

The export process also writes an updated transcript table copy under `transcript_tables/` with a `derived_file` column.

## Reliability Support

Transcription reliability selection writes a workbook named `transcription_reliability_samples.xlsx` with selection and all-transcript sheets. Evaluation and reselection are separate command paths. Command pages should document those details rather than repeating them across the module.

## Boundaries

The implementation assumes that later transcript-table consumers need stable identifiers and predictable workbook structure. It does not treat transcript tables as a permanent database format, and it does not make CHAT export a lossless formatting guarantee.
