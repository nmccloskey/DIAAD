# `cus rates` Usage Guide

Use `diaad cus rates` after `diaad cus analyze` has produced a long sample summary and speaking-time values have been entered.

## Required Inputs

The CU sample summary must contain:

```text
sample_id
coder
paradigm
sv_col
rel_col
cu_col
p_sv
p_rel
cu
```

The speaking-time workbook must contain:

```text
sample_id
speaking_time
```

The speaking-time value is interpreted as seconds. DIAAD converts it to minutes internally.

## Important Settings

| Setting | Default | Effect |
|---|---|---|
| `advanced.cu_samples_filename` | `cu_coding_by_sample_long.xlsx` | Long CU sample summary to read. |
| `advanced.speaking_time_filename` | `speaking_times.xlsx` | Speaking-time workbook to read. |
| `advanced.speaking_time_column` | `speaking_time` | Column containing speaking time in seconds. |
| `advanced.sample_id_column` | `sample_id` | Identifier used to merge CU summaries and speaking time. |

## Output Columns

The rates output keeps a compact set of columns:

```text
sample_id
coder
paradigm
sv_col
rel_col
cu_col
speaking_time
speaking_minutes
cu_per_min
p_sv_per_min
p_rel_per_min
```

Rates are rounded to three decimal places.

## Duplicate Or Missing Speaking Times

If the speaking-time workbook contains duplicate sample IDs, DIAAD sums their speaking-time values before merging.

If a CU sample summary row has no matching speaking-time value, rate columns for that row remain missing.

## Common Problems

If the command cannot find the sample summary, check that `cus analyze` has run and that `advanced.cu_samples_filename` matches the workbook name.

If rates are much smaller or larger than expected, confirm that speaking-time values were entered in seconds, not minutes.

If some samples have missing rates, check sample identifier consistency between the CU summary and the speaking-time workbook.
