# Transcript Preprocessing, Tabularization, Auto-Tabularization, and CHAT Export Implementation Notes

Transcript preprocessing is implemented across the transcript readers, table builders, detabularization helpers, and run-context prerequisite logic.

## Source Anchors

Primary sources:

- `src/diaad/transcripts/cha_files.py`
- `src/diaad/transcripts/transcript_tables.py`
- `src/diaad/transcripts/detabularization.py`
- `src/diaad/core/run_context.py`
- `src/diaad/cli/dispatch.py`

Relevant tests:

- `tests/test_transcripts/test_transcript_tables.py`
- `tests/test_transcripts/test_detabularization.py`
- `tests/test_core/test_run_context.py`

## CHAT Loading

`read_cha_files()` recursively reads `.cha` files from the configured input directory with `pylangacq.Reader.from_files()`. It stores each file under a path that is portable relative to the configured input directory.

When the run context loads CHAT files, it excludes directories named by the configured reliability directory name. The packaged default is `reliability`.

If `project.shuffle_samples` is true, CHAT file order is shuffled before table construction.

## Transcript Table Structure

`tabularize_transcripts()` writes the configured transcript table filename under:

```text
transcript_tables/
```

By default, that file is:

```text
transcript_tables/transcript_tables.xlsx
```

It writes:

- `samples`: one row per source transcript;
- `utterances`: one row per utterance;
- `metadata_mismatches`: diagnostic rows for unresolved metadata fields.

Default sample columns include `sample_id`, `file`, `file_ext`, `file_dir`, `input_order`, `shuffled_order`, configured metadata fields, and `metadata_mismatch`.

Default utterance columns include `sample_id`, `utterance_id`, `position`, `position_sub`, `speaker`, `utterance`, and `comment`.

Sample IDs use the configured sample identifier column name and padded values such as `S001`. Utterance IDs use the configured utterance identifier column name and padded values such as `U001`. Utterance numbering restarts within each sample, so downstream joins should treat the sample and utterance identifiers together when utterance-level uniqueness matters.

## Prerequisite Behavior

`src/diaad/cli/dispatch.py` marks transcript-table-requiring commands. If any such command is requested and `transcripts tabularize` is not part of the same run, `RunContext.ensure_transcript_tables()` searches the configured input directory and the current run output directory for the configured transcript table filename.

If no table is found:

- with `auto_tabularize: false`, DIAAD raises an error;
- with `auto_tabularize: true`, DIAAD reads available CHAT files and writes transcript tables in the current run output directory.

## Detabularization

`detabularize_transcripts()` finds the configured transcript table, loads the `samples` and `utterances` sheets, assigns derived CHAT filenames, and writes `.cha` files under:

```text
chat_files/
```

It sorts utterances by `position` and `position_sub` when those columns are present. It requires `speaker` and `utterance` in the utterance sheet. It preserves comments as `%com` lines.

The updated transcript table copy includes a `derived_file` column in the sample sheet.

## Read Next

- Transcript tabularization feature: `docs/manual/03_features/01_transcript_tabularization.md`
- Transcripts implementation notes: `docs/manual/04_modules/01_transcripts/04_implementation_notes.md`
- File discovery implementation notes: `docs/manual/05_functionalities/09_configured_filenames_file_discovery_input_selection/04_implementation_notes.md`
