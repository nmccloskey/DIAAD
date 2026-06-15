# `turns evaluate` Usage Guide

Use `diaad turns evaluate` after primary and reliability coders have completed DCT strings for the selected reliability material.

## File Discovery

The command looks in the configured input directory and current run output directory for these exact filenames:

```text
conversation_turns_template.xlsx
conversation_turns_reliability_template.xlsx
```

If multiple files with the same expected name are available, exact-file discovery behavior applies. Keep the evaluation input directory clean, or move the intended workbooks into a dedicated subdirectory.

## Required Fields

The primary and reliability workbooks must include:

```text
sample_id
turns
```

`sample_id` may be replaced by the configured `advanced.sample_id_column`. If the configured sample identifier is missing but a `group` column is present, DIAAD can treat `group` as the sample identifier.

`session` and `bin` are added as blank fields if missing, but they are important for meaningful comparison. The reliability unit is the sample/session/bin row.

## Normalization And Pairing

DIAAD trims sample, session, bin, and turns values to strings. Duplicate sample/session/bin rows are dropped after a warning, keeping the first occurrence.

Primary and reliability rows are merged with an outer join. Rows that appear in only one workbook are retained with a blank turn string on the missing side, and the coverage section in the report helps identify those gaps.

## Output Workbook

The `counts` sheet compares digit counts by participant within each sample/session/bin unit. It includes:

```text
count_main
count_rel
perc_agmt
```

`perc_agmt` is the smaller count divided by the larger count, expressed as a percentage. If both counts are zero, the value is treated as full agreement for that participant target.

The `sequences` sheet compares full turn strings with Levenshtein distance and similarity. The `samples` sheet aggregates count and sequence summaries to the sample level.

## Report And Alignments

The plain-text report includes coverage, mean count percent agreement, ICC(2,1) for participant count targets, variance summaries, and sequence-similarity bands.

The `global_alignments/` folder contains one text file per sample/session/bin comparison when alignment output can be produced. If one side is blank, the alignment file records that the alignment is unavailable.

## Common Problems

If the command cannot find files, check that the filenames are exact and that the workbooks are in the configured input directory or current run output directory.

If agreement is lower than expected, inspect whether coders disagreed about speaker labels, turn boundaries, dot-marker syntax, or the session/bin row structure.

If rows seem to disappear, check for duplicate sample/session/bin keys in either workbook.

## Read Next

- `turns reselect` quickstart: `docs/manual/04_modules/07_digital_conversational_turns/05_commands/03_reselect/01_quickstart.md`
- `turns analyze` quickstart: `docs/manual/04_modules/07_digital_conversational_turns/05_commands/04_analyze/01_quickstart.md`
- Generated Example I/O: `docs/manual/03_features/04_generated_example_io.md`
