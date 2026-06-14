# `transcripts select` Implementation Notes

`transcripts select` dispatches to `select_transcription_reliability_samples()`.

## Dispatch Path

The main path is:

1. `src/diaad/cli/commands.py` registers `transcripts select`.
2. `src/diaad/cli/dispatch.py` marks it as a CHAT-required command.
3. `src/diaad/core/run_context.py` loads CHAT files and threads `reliability_fraction` and the configured transcript table filename.
4. `src/diaad/core/run_wrappers.py` calls `select_transcription_reliability_samples()`.
5. `src/diaad/transcripts/transcription_reliability_selection.py` writes the selection workbook and optional blank CHAT headers.

## Sample Frame Resolution

The implementation calls `find_transcript_table()` with:

```text
project.input_dir
current run output directory
```

and the configured transcript table filename. If a matching workbook is found, the `samples` sheet is used. If none is found, the command falls back to sample rows built from loaded CHAT files.

## Output Workbook

The selection workbook is written to:

```text
transcription_reliability_selection/transcription_reliability_samples.xlsx
```

It contains:

- `reliability_selection`
- `all_transcripts`

The `all_transcripts` sheet includes `selected_for_reliability`.

## Selection Behavior

Subset size is calculated by the shared sampling helper used across DIAAD reliability and subset commands. The command samples from the active sample frame and sorts selected rows by `file` when that column is present.

The run context seeds Python's random generator from `project.random_seed` before command dispatch.

## Relevant Sources

- `src/diaad/transcripts/transcription_reliability_selection.py`
- `src/diaad/coding/utils/sampling.py`
- `src/diaad/core/run_context.py`
- `src/diaad/core/run_wrappers.py`
