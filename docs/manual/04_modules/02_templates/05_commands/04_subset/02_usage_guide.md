# `templates subset` Usage Guide

Use `diaad templates subset` when you need a simple randomized sample subset from an existing workbook.

## Input Workbook Rule

This command intentionally accepts any `.xlsx` filename, but the input directory must contain exactly one Excel workbook. If the directory contains zero workbooks or multiple workbooks, DIAAD stops instead of guessing.

A common layout is:

```text
diaad_data/input/sample_subset/
  sample_subset_input.xlsx
```

with:

```yaml
input_dir: diaad_data/input/sample_subset
```

## Required Sheet

The input workbook must contain a `samples` sheet. The sheet must include the configured sample identifier column, which defaults to `sample_id`.

Duplicate sample IDs are collapsed to one output status row per sample.

## Plain Subset Mode

If the input `samples` sheet does not contain `exclude`, DIAAD treats all unique sample IDs as eligible. It writes `selected` and `excluded` columns in the output, with `excluded` set to `0`.

## Re-Subset Mode

If the input `samples` sheet contains `exclude`, the column must contain only binary values: `0` or `1`.

DIAAD calculates the target subset size from the full sample set, but samples marked `exclude == 1` are not eligible for selection. If duplicate rows for a sample have mixed exclude values, the sample is treated as excluded when any duplicate row is excluded.

## Selection Size

Selection size uses:

```yaml
reliability_fraction: 0.2
random_seed: 99
```

The selected count is the ceiling of `reliability_fraction * number_of_samples`, with a minimum of one. If too few eligible samples remain after exclusions, DIAAD selects as many as possible and logs a warning.

## Common Problems

If the command reports multiple input workbooks, point `input_dir` at a narrower folder or remove extra `.xlsx` files.

If the command rejects `exclude`, check for blanks, words, or values other than `0` and `1`.

If selected rows are fewer than expected, check how many samples are marked excluded.
