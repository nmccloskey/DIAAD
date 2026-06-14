# `templates utterances` Usage Guide

Use `diaad templates utterances` when the coding unit is an utterance and DIAAD does not already provide a specialized command for the coding scheme.

## Before Running

Create or provide transcript tables first. The `utterances` sheet must contain the configured sample and utterance identifier columns plus `utterance`. The `samples` sheet is used to attach a stimulus column when one is configured and present.

Relevant configuration:

```yaml
reliability_fraction: 0.2
num_coders: 2
stimulus_column: narrative
```

`num_coders` controls coder assignment. If it is `0`, DIAAD still creates a `coder_id` column, but the value is blank.

## Important Settings

| Setting | Default | Effect |
|---|---|---|
| `project.reliability_fraction` | `0.2` | Fraction of samples selected into the reliability workbook. |
| `project.num_coders` | `0` | Number of coder IDs to assign. |
| `project.random_seed` | `99` | Seed for coder assignment and reliability subset sampling. |
| `project.stimulus_column` | `''` | Optional sample-level column to copy into the template as `stimulus`. |
| `advanced.transcript_table_filename` | `transcript_tables.xlsx` | Transcript table workbook to discover. |
| `advanced.sample_id_column` | `sample_id` | Sample identifier column. |
| `advanced.utterance_id_column` | `utterance_id` | Utterance identifier column. |
| `advanced.auto_blind` | `false` | Whether supported coding templates should use configured blinding columns. |
| `advanced.blind_columns` | `[sample_id]` | Identifier columns to blind when `auto_blind` is enabled. |

## Coder Assignment And Reliability Rows

DIAAD assigns primary coder IDs by sample so that all rows from a sample stay together. When multiple coders are configured, the reliability workbook contains selected samples assigned to an alternate coder when possible.

The reliability subset is sample-preserving. If a sample is selected, all template rows for that sample appear in the reliability workbook.

## Optional Blinding

When `advanced.auto_blind` is true and configured blind columns are present, DIAAD writes blinded template workbooks and a codebook. Keep the codebook protected, because it maps blinded values back to raw identifiers.

Software blinding does not replace a project-level privacy or coder-masking plan. It only changes configured identifier columns in the generated workbook.

## Common Problems

If the command cannot find a transcript table, check the configured filename and remove duplicate copies under the active input tree. See Exact file name matching (`docs/manual/03_features/03_exact_file_name_matching.md`).

If no `stimulus` column appears, check that `project.stimulus_column` is set and that the named column exists in the transcript table `samples` sheet.

If the reliability workbook has fewer rows than expected, remember that selection happens by sample, not by individual utterance row.
