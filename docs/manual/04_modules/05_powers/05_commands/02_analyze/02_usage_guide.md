# `powers analyze` Usage Guide

Use `diaad powers analyze` after POWERS coding has been completed and checked.

## Required Workbook

DIAAD searches the input directory and current run output directory for the configured POWERS coding filename. The default is:

```text
powers_coding.xlsx
```

The current implementation reads the first sheet in that workbook. DIAAD-generated primary workbooks place `utterance_coding` first.

## Required Coding Columns

The analysis path requires the configured sample identifier, `speaker`, and `turn_type` to produce the full set of summaries. Default sample identifier:

```text
sample_id
```

Valid turn types for numbering are:

```text
T
MT
ST
NV
```

Blank or invalid `turn_type` values inherit the prior turn label when possible; if the first value is invalid, DIAAD marks it `X`.

## Summarized Metrics

The analysis summaries aggregate these utterance-level numeric fields when present:

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

The dialog sheet also summarizes collaborative repair as `num_repairs` and `prop_repairs` when `collab_repair` is present.

## Section E Boundary

The `section_e` sheet created by `powers files` is not included in the current analysis output. Those fields are sample-level note or descriptor fields rather than operationalized utterance-level metrics in the current DIAAD summary path.

## Output Sheets

`powers_analysis.xlsx` can include:

| Sheet | Contents |
|---|---|
| `Utterances` | Utterance-level data with `turn_label` inserted after `turn_type`. |
| `Turns` | Counts summed by sample, speaker, and turn label. |
| `Speakers` | Counts, turn-type counts, mean turn length, and selected ratios by sample and speaker. |
| `Dialogs` | Dialog/sample-level count sums plus repair summaries. |

## Common Problems

If analysis fails, check that the workbook contains `turn_type`, `speaker`, and the configured sample identifier column.

If turn labels look strange, inspect blank or invalid `turn_type` values and the row order of the workbook.

If a metric is missing from summaries, check that the source column is present and numeric in the utterance-level coding sheet.
