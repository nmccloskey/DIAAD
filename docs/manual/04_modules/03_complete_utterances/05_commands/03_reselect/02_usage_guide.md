# `cus reselect` Usage Guide

Use `diaad cus reselect` when additional CU reliability material is needed after a prior reliability round.

## Input Discovery

The command searches recursively under `project.input_dir` for files matching:

```text
*cu_coding.xlsx
*cu_reliability_coding.xlsx
```

It pairs coding and reliability files using configured metadata fields. If metadata fields are not configured, matching falls back to file stems.

## Selection Behavior

DIAAD reads the original CU coding workbook and collects sample IDs that already appear in matched reliability workbooks. It then samples from unused sample IDs.

The target sample count uses:

```yaml
reliability_fraction: 0.2
random_seed: 99
```

If fewer unused samples remain than the target count, DIAAD selects what is available.

## Output Shape

The reselected workbook keeps the original coding columns through the first comment-like boundary column, then preserves reliability-side columns from the prior reliability template when available. If expected reliability admin columns such as `c3_id` or `c3_comment` are absent, DIAAD adds them.

## Common Problems

If no pairs are found, check the input directory and metadata-field matching.

If no samples are selected, all samples may already be represented in prior reliability files.

If the output columns are not what you expected, compare the original coding workbook with the prior reliability workbook. Reselection uses those files as the shape template.
