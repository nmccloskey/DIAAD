# `words analyze` Usage Guide

Use `diaad words analyze` after word-count coding has been completed and checked.

## Required Workbook

DIAAD searches the input directory and current run output directory for the configured word-counting filename. The default is:

```text
word_counting.xlsx
```

The workbook must contain the configured sample identifier column and word-count column. Defaults are:

```text
sample_id
word_count
```

Coder identifier columns (`coder_id`) are not required.


## Word-Count Coercion

DIAAD coerces the configured word-count column to numeric values. Numeric values are summarized. Blank values, `NA`, and other nonnumeric entries are treated as missing for the summary.

Rows whose speaker label appears in `project.exclude_speakers` are dropped before summary when a `speaker` column is present.

## Outputs

`word_counting_by_utterance.xlsx` contains cleaned utterance-level word-count data with administrative coding columns removed.

`word_counting_by_sample.xlsx` contains one row per sample with:

```text
no_utt_coded
no_utt_missing
total_words
mean_words_per_utt
sd_words_per_utt
min_words_per_utt
max_words_per_utt
```

Summary values are rounded to three decimal places.

## Blinding Behavior

If a word-counting-stage blind codebook is available, the analysis path can reconnect sample identifiers before writing outputs. If analysis-stage blinding is configured, DIAAD can then write blinded analysis outputs plus `word_count_analysis_blind_codebook.xlsx` and `word_count_analysis_blinding_diagnostics.xlsx`.

## Common Problems

If no workbook is found, check `advanced.word_count_filename` and the exact file name in the input directory.

If totals are lower than expected, check for `NA`, blank, or nonnumeric word-count entries and confirm that excluded speaker labels are configured correctly.

If outputs contain blinded sample IDs when you expected raw IDs, check the available blind codebook and the current `advanced.auto_blind` settings.
