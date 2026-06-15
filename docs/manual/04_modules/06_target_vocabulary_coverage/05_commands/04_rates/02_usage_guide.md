# `vocab rates` Usage Guide

Use `diaad vocab rates` after `diaad vocab analyze` has produced one or more analysis workbooks.

## Required Input

DIAAD searches the input directory and current run output directory for workbooks matching:

```text
target_vocab_data_*.xlsx
```

It reads the `summary` sheet from each matching workbook. Each summary must contain the configured sample identifier column and:

```text
speaking_time
```

`speaking_time` is interpreted as seconds.

## Rate Numerators

DIAAD infers per-minute numerators from numeric summary columns. It excludes:

```text
sample_id
narrative
speaking_time
speaking_minutes
source_file
lexicon_coverage
accuracy_pwa_percentile
accuracy_control_percentile
efficiency_pwa_percentile
efficiency_control_percentile
core_tokens_per_min
columns ending with _per_min
```

This means count-like fields such as `num_tokens`, `num_base_forms_produced`, and `num_core_token_matches` can receive additional per-minute columns, while coverage, percentile, and existing rate fields are not rate-normalized again.

## Output Columns

The rates output keeps a compact order:

```text
sample_id
narrative
source_file
count-like numerator columns
speaking_time
speaking_minutes
core_tokens_per_min
per-minute columns
percentile columns
```

## Common Problems

If no analysis workbook is found, run `vocab analyze` first or check where the analysis workbook was placed.

If rates are much smaller or larger than expected, confirm that `speaking_time` values in the analysis summary are seconds, not minutes.

If duplicate samples appear, check whether multiple analysis workbooks were intentionally included. The `source_file` column identifies where each row came from.
