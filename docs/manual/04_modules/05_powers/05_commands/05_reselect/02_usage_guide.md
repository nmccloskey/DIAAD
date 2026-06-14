# `powers reselect` Usage Guide

Use `diaad powers reselect` when additional or replacement POWERS reliability material is needed after a prior reliability round.

## Input Discovery

The command searches recursively under `project.input_dir` for files matching the configured primary and reliability filenames. Defaults are:

```text
*powers_coding.xlsx
*powers_reliability_coding.xlsx
```

It pairs coding and reliability files using configured metadata fields. If metadata fields are not configured, matching falls back to file stems.

## Selection Behavior

DIAAD reads the original POWERS coding workbook and collects sample IDs that already appear in matched reliability workbooks. It then samples from unused sample IDs.

The target sample count uses:

```yaml
reliability_fraction: 0.2
random_seed: 99
```

If fewer unused samples remain than the target count, DIAAD selects what is available.

## Output Behavior

The reselected workbook keeps the original leading columns through the first comment-like boundary column, then preserves useful reliability-template columns when available.

Manual POWERS coding fields are cleared so the selected rows can be coded fresh. If `project.automate_powers` is true and NLP support is available, DIAAD reapplies first-pass automation for:

```text
speech_units
filled_pauses
content_words
num_nouns
tagged_utterance
```

Review `coder_id` and automated values before distributing the workbook.

## Section E Boundary

Reselection operates on the utterance-level coding sheet. It does not create or update the Section E sheet.

## Common Problems

If no pairs are found, check the input directory and metadata-field matching.

If no samples are selected, all samples may already be represented in prior reliability files.

If automated first-pass fields are blank, check `project.automate_powers`, the installed NLP dependencies, and `advanced.spacy_model_name`.
