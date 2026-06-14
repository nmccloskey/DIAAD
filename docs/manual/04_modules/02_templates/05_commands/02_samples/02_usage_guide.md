# `templates samples` Usage Guide

Use `diaad templates samples` when the coding unit is a whole sample, a sample-by-bin unit, or another sample-level row that does not need every transcript utterance.

## Before Running

Create or provide transcript tables first. The command reads the `samples` sheet and requires the configured sample identifier column.

Set the number of bins if each sample should be expanded into repeated rows:

```yaml
num_bins: 4
```

Set coder count and reliability fraction if the template should organize primary and reliability assignments:

```yaml
num_coders: 2
reliability_fraction: 0.2
```

## Important Settings

| Setting | Default | Effect |
|---|---|---|
| `project.num_bins` | `4` | Number of bin rows created for each sample. |
| `project.num_coders` | `0` | Number of coder IDs to assign. |
| `project.reliability_fraction` | `0.2` | Fraction of samples selected into the reliability workbook. |
| `project.random_seed` | `99` | Seed for coder assignment and reliability subset sampling. |
| `project.stimulus_column` | `''` | Optional sample-level column copied into output as `stimulus`. |
| `advanced.sample_id_column` | `sample_id` | Sample identifier column. |
| `advanced.auto_blind` | `false` | Whether supported coding templates should blind configured columns. |
| `advanced.blind_columns` | `[sample_id]` | Identifier columns to blind when `auto_blind` is enabled. |

## Rows And Bins

The command starts with unique sample rows from the transcript table. It then expands each sample into bin labels from `1` through `num_bins`.

For example, `num_bins: 4` creates four rows per sample before coder assignment. The `bin` column is a scaffold for project-defined coding units; DIAAD does not impose a coding meaning on the bins.

## Coder Assignment And Reliability Rows

Primary coder IDs are assigned by sample. The reliability workbook selects samples and assigns them to an alternate coder when multiple coders are configured.

Like utterance templates, sample-template reliability is sample-preserving. When a sample is selected, its bin rows appear together.

## Common Problems

If too many rows appear, check `project.num_bins`. The default is `4`, so every sample is expanded into four bin rows.

If `coder_id` is blank, set `project.num_coders` to the number of coders you want represented in the workbook.

If downstream work needs a different bin definition, edit the project protocol first. The generated bin labels are structural placeholders, not validated categories.
