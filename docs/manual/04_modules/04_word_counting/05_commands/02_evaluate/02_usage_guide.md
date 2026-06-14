# `words evaluate` Usage Guide

Use `diaad words evaluate` after the primary word-counting workbook and reliability workbook have both been reviewed and completed.

## Required Workbooks

DIAAD searches the input directory and current run output directory for exactly one:

```text
word_counting.xlsx
word_count_reliability.xlsx
```

The files must contain compatible sample and utterance identifiers. The default identifiers are `sample_id` and `utterance_id`.

## Comparison Behavior

The command merges primary and reliability rows by the configured sample and utterance identifier columns.

Only rows with numeric word counts in both files are evaluated. Rows marked `NA`, blank rows, and other nonnumeric values are excluded from the paired reliability comparison.

## Agreement Rule

For each paired utterance, DIAAD marks agreement when either condition is true:

```text
absolute difference <= 1 word
percent similarity >= 85%
```

The results workbook includes:

```text
word_count_org
word_count_rel
abs_diff
perc_diff
perc_sim
agmt
```

## Report Contents

The text report includes:

- coverage in the primary coding file;
- number of paired utterances;
- number of utterances marked in agreement;
- mean absolute difference;
- mean percent difference;
- mean percent similarity;
- ICC(2,1);
- variance diagnostics.

## Common Problems

If paired utterances are fewer than expected, check that sample and utterance identifiers match across the two workbooks.

If rows disappear from the comparison, check whether one workbook contains blank, `NA`, or nonnumeric word-count values.

If ICC is missing or surprising, inspect the paired values and variance diagnostics. ICC can be undefined when there are too few paired rows or too little variance.
