# `transcripts reselect` Usage Guide

Use `diaad transcripts reselect` when a transcription reliability sample needs replacement or when a project needs an additional reliability round that avoids previously selected samples.

## Before Running

Move or copy the prior selection workbook into the active input directory. DIAAD searches recursively for files matching:

```text
*transcription_reliability_samples.xlsx
```

The workbook should contain:

- `all_transcripts`
- `reliability_selection`

The preferred way to identify previously selected rows is the `selected_for_reliability` column in `all_transcripts`. If that column is absent, DIAAD falls back to files listed in `reliability_selection`.

## Reliability Fraction

The target replacement sample count is based on the same setting used for selection:

```yaml
reliability_fraction: 0.2
```

DIAAD calculates the target size from all transcripts in the prior workbook, then samples from candidates that were not previously selected.

## Output

For each eligible prior selection workbook, DIAAD writes:

```text
reselected_transcription_reliability/reselected_<prior filename>
```

The usual output filename is:

```text
reselected_transcription_reliability_samples.xlsx
```

The output sheet is `reselected_reliability`.

## Common Problems

If no output appears, check that the prior selection workbook is under `project.input_dir` and matches the expected filename pattern.

If a workbook is skipped, check that it contains both required sheets and a `file` column.

If fewer rows are selected than expected, all remaining candidates may have been exhausted. DIAAD logs a warning when it cannot meet the requested fraction.
