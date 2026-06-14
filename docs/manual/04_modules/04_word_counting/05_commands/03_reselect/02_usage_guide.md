# `words reselect` Usage Guide

Use `diaad words reselect` when an earlier word-count reliability round needs replacement or additional material.

## Input Discovery

The command searches recursively under `project.input_dir` for files matching:

```text
*word_counting.xlsx
*word_count_reliability.xlsx
```

It pairs coding and reliability files using configured metadata fields. If metadata fields are not configured, matching falls back to file stems.

## Selection Behavior

DIAAD reads the original word-counting workbook and collects sample IDs that already appear in matched reliability workbooks. It then samples from unused sample IDs.

The target sample count uses:

```yaml
reliability_fraction: 0.2
random_seed: 99
```

If fewer unused samples remain than the target count, DIAAD selects what is available.

## Output Shape

The reselected workbook keeps the original columns through the first comment-like boundary column, then preserves reliability-template columns from the prior reliability workbook when available.

If `wc_rel_com` is absent, DIAAD adds it. The command also recomputes a first-pass `word_count` value from the utterance text for selected rows.

## Common Problems

If no pairs are found, check the input directory and metadata-field matching.

If no samples are selected, all samples may already be represented in prior reliability files.

If selected rows have unexpected first-pass counts, review them before distribution. The generated counts are starting values for human review, not final adjudicated counts.
