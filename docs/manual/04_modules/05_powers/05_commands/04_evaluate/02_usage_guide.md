# `powers evaluate` Usage Guide

Use `diaad powers evaluate` after primary and reliability POWERS workbooks have both been reviewed and completed.

## Required Workbooks

DIAAD searches the input directory and current run output directory for exactly one:

```text
powers_coding.xlsx
powers_reliability_coding.xlsx
```

The files must contain compatible sample and utterance identifiers. The default identifiers are `sample_id` and `utterance_id`.

## Evaluated Metrics

Continuous or count-like metrics:

```text
speech_units
content_words
num_nouns
filled_pauses
circumlocutions
sem_paras
phon_errs
neologisms
lg_pauses
```

Categorical metrics:

```text
turn_type
collab_repair
```

For `collab_repair`, the categorical summary compares whether a repair value is present, rather than treating every repair label as a distinct categorical code.

Section E fields are not evaluated by this command.

## Output Workbook

`powers_reliability_results.xlsx` can contain:

| Sheet | Contents |
|---|---|
| `merged` | Paired utterance rows with original and reliability values plus row-level continuous differences. |
| `continuous_summary` | ICC, exact agreement, within-one-count agreement, mean differences, missingness, and variance diagnostics for continuous metrics. |
| `categorical_summary` | Percent agreement, kappa, and variance diagnostics for categorical metrics. |

## Report Contents

The text report includes:

- coverage in the primary coding file;
- paired utterance count;
- continuous metric summaries;
- categorical metric summaries;
- warnings for sparse or low-variance count metrics.

## Common Problems

If paired utterances are fewer than expected, check that sample and utterance identifiers match across the two workbooks.

If continuous metrics are missing, confirm that both workbooks contain numeric values for those columns.

If ICC is low despite high agreement, inspect the `ICC_warning`, exact agreement, within-one-count agreement, and distribution diagnostics. Low variance can make ICC hard to interpret.
