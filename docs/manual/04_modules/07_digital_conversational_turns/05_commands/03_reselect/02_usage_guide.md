# `turns reselect` Usage Guide

Use `diaad turns reselect` when an initial DCT reliability subset has already been coded and the project needs additional or replacement reliability material without repeating already-used samples.

## Input Placement

Unlike some commands that also inspect the current output directory, DCT reselection searches the configured input directory recursively. Put the prior primary and reliability workbooks under `input_dir` before running the command.

Expected filename patterns are:

```text
*conversation_turns_template.xlsx
*conversation_turns_reliability*.xlsx
```

## Pairing Behavior

DIAAD first tries shared reliability-pair discovery using configured metadata fields. If no metadata-based reliability mates are found, it falls back to filename-based pairing with one primary conversation-turns template and all matching reliability workbooks.

## Selection Behavior

The command reads all sample identifiers that already appear in prior reliability files. It then selects new sample identifiers from the primary workbook, excluding any already used.

The selection size is based on:

```text
project.reliability_fraction
```

The random selection is reproducible when `project.random_seed` is set.

## Output Shape

The reselected workbook preserves the primary workbook's session and bin structure for selected samples. If a `turns` column exists, it is cleared so the workbook is ready for fresh reliability coding.

The usual filename is:

```text
reselected_conversation_turns_reliability_template.xlsx
```

If the primary file has an additional stem prefix, the helper preserves that prefix before `reselected_`.

## Common Problems

If no output appears, check whether all samples in the primary file have already appeared in prior reliability files.

If the wrong workbooks are paired, move unrelated DCT workbooks out of the configured input directory or adjust metadata fields so filenames can be matched more specifically.

If the output lacks the expected identifier column, confirm `advanced.sample_id_column`.

## Read Next

- `turns evaluate` usage guide: `docs/manual/04_modules/07_digital_conversational_turns/05_commands/02_evaluate/02_usage_guide.md`
- Generated Example I/O: `docs/manual/03_features/04_generated_example_io.md`
