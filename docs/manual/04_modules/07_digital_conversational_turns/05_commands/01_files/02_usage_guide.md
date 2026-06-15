# `turns files` Usage Guide

Use `diaad turns files` when sample identifiers are ready and a project needs blank DCT coding material.

## Input Basis

The file-generation command uses transcript tables only to obtain the sample list. It reads the sample sheet from the configured transcript table workbook and keeps unique values from the configured sample identifier column.

The default sample identifier is:

```text
sample_id
```

If the project uses a different identifier column, set `advanced.sample_id_column`.

## Template Shape

The primary template is built around these columns:

```text
sample_id
coder_id
session
bin
turns
```

`session` is created blank. `bin` is populated from the configured number of bins. With the default `project.num_bins: 4`, each sample receives bin labels `1`, `2`, `3`, and `4`.

## Important Settings

| Setting | Role |
|---|---|
| `project.num_bins` | Number of bin rows to create for each sample. |
| `project.num_coders` | Controls coder assignment in the primary and reliability templates. |
| `project.reliability_fraction` | Fraction of samples selected for the reliability template. |
| `project.random_seed` | Makes reliability selection and blinding reproducible. |
| `advanced.sample_id_column` | Names the sample identifier column in transcript tables and output templates. |
| `advanced.transcript_table_filename` | Names the transcript table workbook. |
| `advanced.auto_blind` | Applies configured coding blinding when enabled. |
| `advanced.blind_columns` | Identifies columns to encode during automatic blinding. |

## Coder Assignment

When `project.num_coders` is `0`, `coder_id` is left blank. When it is `1`, the primary template uses coder `1`. With two or more coders, DIAAD assigns primary coding material by sample and builds reliability material with alternate coder assignments.

Reliability selection is sample-preserving. When a sample is selected for reliability coding, all of its generated bin rows are included.

## Turn-String Syntax

Use digits and optional dot markers only:

```text
0.1..23.0.12
```

The parser reads each digit as a separate speaker code. It does not interpret `10` as participant ten; it reads that as `1` followed by `0`.

One dot after a digit is counted as `mark1`, and two dots after a digit are counted as `mark2`. Dot markers do not change the speaker turn count or the transition sequence, but they are summarized by the analysis command.

## Common Problems

If no template is written, confirm that transcript tables exist or can be generated from the configured input directory.

If the output has too many or too few rows, check `project.num_bins`, `project.num_coders`, and `project.reliability_fraction`.

If identifiers are unexpectedly numeric or masked, check whether automatic blinding is enabled and whether the generated codebook was written.

## Read Next

- `turns evaluate` quickstart: `docs/manual/04_modules/07_digital_conversational_turns/05_commands/02_evaluate/01_quickstart.md`
- `turns analyze` quickstart: `docs/manual/04_modules/07_digital_conversational_turns/05_commands/04_analyze/01_quickstart.md`
- Digital Conversational Turns research context: `docs/manual/04_modules/07_digital_conversational_turns/03_research_context.md`
- Exact file-name matching: `docs/manual/03_features/03_exact_file_name_matching.md`
