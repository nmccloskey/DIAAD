# `powers rates` Usage Guide

Use `diaad powers rates` after `diaad powers analyze` has produced dialog-level POWERS summaries and speaking-time values have been entered.

## Required Inputs

DIAAD searches the input directory and current run output directory for POWERS analysis workbooks matching:

```text
*powers*analysis*.xlsx
```

It reads the `Dialogs` sheet from each matching workbook.

The speaking-time workbook must contain:

```text
sample_id
speaking_time
```

The speaking-time value is interpreted as seconds. DIAAD converts it to minutes internally.

## Important Settings

| Setting | Default | Effect |
|---|---|---|
| `advanced.speaking_time_filename` | `speaking_times.xlsx` | Speaking-time workbook to read. |
| `advanced.speaking_time_column` | `speaking_time` | Column containing speaking time in seconds. |
| `advanced.sample_id_column` | `sample_id` | Identifier used to merge POWERS summaries and speaking time. |

## Rate Numerators

DIAAD infers per-minute numerators from numeric columns in the combined dialog summaries. It excludes:

```text
sample_id
source_file
speaking_time
speaking_minutes
columns beginning with prop_
columns beginning with ratio_
```

This means count-like dialog columns such as `speech_units_sum`, `content_words_sum`, and `num_repairs` can receive per-minute columns, while proportions and ratios are not rate-normalized again.

## Section E Boundary

Because Section E is not included in `powers analyze` dialog summaries, Section E fields are not converted to rates.

## Common Problems

If no analysis workbook is found, run `diaad powers analyze` first or check where the analysis workbook was placed.

If duplicate sample IDs appear across multiple analysis workbooks, DIAAD retains all rows and adds `source_file` so the source workbook remains visible.

If rates are much smaller or larger than expected, confirm that speaking-time values were entered in seconds, not minutes.
