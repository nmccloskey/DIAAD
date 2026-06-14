# `words rates` Usage Guide

Use `diaad words rates` after `diaad words analyze` has produced a sample-level word-count summary and speaking-time values have been entered.

## Required Inputs

The word-count sample summary must contain:

```text
sample_id
total_words
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
| `advanced.wc_samples_filename` | `word_counting_by_sample.xlsx` | Word-count sample summary to read. |
| `advanced.speaking_time_filename` | `speaking_times.xlsx` | Speaking-time workbook to read. |
| `advanced.speaking_time_column` | `speaking_time` | Column containing speaking time in seconds. |
| `advanced.sample_id_column` | `sample_id` | Identifier used to merge word-count summaries and speaking time. |

## Output Columns

The rates output keeps word-count sample summary columns when available and adds:

```text
speaking_time
speaking_minutes
total_words_per_min
```

Rates are rounded to three decimal places.

## Duplicate Or Missing Speaking Times

If the speaking-time workbook contains duplicate sample IDs, DIAAD sums their speaking-time values before merging.

If a word-count summary row has no matching speaking-time value, rate columns for that row remain missing.

## Common Problems

If the command cannot find the sample summary, check that `words analyze` has run and that `advanced.wc_samples_filename` matches the workbook name.

If rates are much smaller or larger than expected, confirm that speaking-time values were entered in seconds, not minutes.

If some samples have missing rates, check sample identifier consistency between the word-count summary and the speaking-time workbook.
