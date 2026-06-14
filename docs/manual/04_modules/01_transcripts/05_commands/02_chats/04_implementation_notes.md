# `transcripts chats` Implementation Notes

`transcripts chats` is the user-facing command for transcript detabularization.

## Dispatch Path

The main path is:

1. `src/diaad/cli/commands.py` registers `transcripts chats`.
2. `src/diaad/cli/dispatch.py` marks it as requiring transcript tables.
3. `src/diaad/core/run_context.py` ensures a transcript table is available unless the same run includes `transcripts tabularize`.
4. `src/diaad/core/run_wrappers.py` calls `detabularize_transcripts()`.
5. `src/diaad/transcripts/detabularization.py` writes CHAT files and a derived transcript table copy.

## Transcript Table Prerequisite

The command searches `project.input_dir` and the current run output directory for the configured transcript table filename. The default is `transcript_tables.xlsx`.

If no matching workbook is available and `project.auto_tabularize` is `false`, the run stops. If `auto_tabularize` is `true`, DIAAD can create transcript tables from available CHAT inputs first.

## Required Columns

The `utterances` sheet must contain the configured sample identifier column plus `speaker` and `utterance`. The command uses `position` and `position_sub` for sorting when those columns are present.

Rows without speaker content are skipped during export.

## Output Behavior

The implementation writes:

```text
chat_files/*.cha
transcript_tables/<configured transcript table filename>
```

The updated workbook includes `derived_file` on the `samples` sheet. The `derived_file` column is excluded when future derived filenames are constructed.

## Relevant Sources

- `src/diaad/transcripts/detabularization.py`
- `src/diaad/metadata/discovery.py`
- `src/diaad/core/run_context.py`
- `src/diaad/core/run_wrappers.py`
- `tests/test_transcripts/test_detabularization.py`
