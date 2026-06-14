# `transcripts select` Usage Guide

Use `diaad transcripts select` before a transcription reliability round. The command creates an auditable selection workbook and, when possible, blank CHAT files for secondary transcription.

## Input Source Priority

The command first searches for the configured transcript table workbook in the input directory and the current run output directory. If a transcript table is found, the `samples` sheet becomes the sample frame.

If no transcript table is found, the command builds the sample frame directly from CHAT files.

This lets a project select reliability samples either after tabularization or directly from raw CHAT files.

## Reliability Fraction

The selection size comes from:

```yaml
reliability_fraction: 0.2
```

DIAAD validates that the fraction is greater than `0` and no greater than `1`. The runtime seeds random selection from `project.random_seed`, so repeated runs with the same inputs and seed should be reproducible.

## Blank Reliability Files

When CHAT data are available, DIAAD writes blank reliability files such as:

```text
P1_picnic_pre_reliability.cha
```

These contain CHAT headers, not completed transcript content. They are intended as setup artifacts for the person doing independent reliability transcription.

If the sample frame comes from a transcript table but no matching CHAT objects are available, DIAAD still writes the workbook and skips blank `.cha` creation.

## What To Inspect

Inspect both workbook sheets:

- `reliability_selection` should contain the intended selected samples.
- `all_transcripts` should contain every eligible sample and the `selected_for_reliability` indicator.

Also confirm that selected files represent the intended sampling frame, especially if metadata fields or input directories changed after tabularization.

## Common Problems

If the selected sample count is unexpected, check `reliability_fraction`, the number of rows in the sample frame, and whether the active transcript table is the intended one.

If blank reliability files are missing, check whether CHAT files were available under `project.input_dir`. A transcript-table-only workflow can select samples without having source CHAT objects to copy headers from.

If metadata columns are missing, create or provide transcript tables with the desired metadata fields before running selection.
