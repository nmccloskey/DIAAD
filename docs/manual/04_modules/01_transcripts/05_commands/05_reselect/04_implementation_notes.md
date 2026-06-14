# `transcripts reselect` Implementation Notes

`transcripts reselect` dispatches to `reselect_transcription_reliability_samples()`.

## Dispatch Path

The main path is:

1. `src/diaad/cli/commands.py` registers `transcripts reselect`.
2. `src/diaad/cli/dispatch.py` dispatches it without requiring CHAT files or transcript tables.
3. `src/diaad/core/run_context.py` threads `input_dir`, `output_dir`, and `reliability_fraction`.
4. `src/diaad/core/run_wrappers.py` calls `reselect_transcription_reliability_samples()`.
5. `src/diaad/transcripts/transcription_reliability_selection.py` reads prior selection workbooks and writes replacement selections.

## Input Discovery

The implementation searches recursively under `project.input_dir` for:

```text
*transcription_reliability_samples.xlsx
```

Each workbook must include `all_transcripts` and `reliability_selection`. The `file` column is required for candidate exclusion.

## Reselection Logic

Prior selected files are identified from `selected_for_reliability` in `all_transcripts` when that column exists. Otherwise, the implementation uses the `file` values in `reliability_selection`.

Candidate rows are all rows in `all_transcripts` whose `file` value was not previously selected. The target count is calculated from `project.reliability_fraction` and the full prior sample frame, then capped by the number of remaining candidates.

## Output

The command writes reselected rows to:

```text
reselected_transcription_reliability/reselected_<prior filename>
```

with sheet name:

```text
reselected_reliability
```

## Relevant Sources

- `src/diaad/transcripts/transcription_reliability_selection.py`
- `src/diaad/coding/utils/sampling.py`
- `src/diaad/core/run_context.py`
- `src/diaad/core/run_wrappers.py`
