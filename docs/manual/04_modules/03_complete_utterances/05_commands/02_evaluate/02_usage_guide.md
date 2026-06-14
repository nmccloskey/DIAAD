# `cus evaluate` Usage Guide

Use `diaad cus evaluate` after primary and reliability CU workbooks have been completed.

## Required Workbooks

DIAAD searches the input directory and current run output directory for exactly one:

```text
cu_coding.xlsx
cu_reliability_coding.xlsx
```

The files must contain compatible sample and utterance identifiers. The default identifiers are `sample_id` and `utterance_id`.

## Supported Coding Schemas

The evaluator supports two reliability comparison modes:

| Mode | Primary workbook columns | Reliability workbook columns |
|---|---|---|
| `primary_vs_reliability` | `sv` and `rel` | `sv` and `rel` |
| `coder2_vs_coder3` | `c2_sv` and `c2_rel` | `c3_sv` and `c3_rel` |

For multi-paradigm workflows, the same pattern is applied with paradigm suffixes such as `sv_AAE`, `rel_AAE`, `c2_sv_AAE`, and `c3_rel_AAE`.

## Output Interpretation

The utterance-level output canonicalizes the comparison to:

```text
c2_sv
c2_rel
c2_cu
c3_sv
c3_rel
c3_cu
agmt_sv
agmt_rel
agmt_cu
```

The sample-level output summarizes counts and percent agreement by sample. The text report includes:

- coverage in the primary coding file;
- utterance-level raw agreement and Cohen's kappa for SV, REL, and CU;
- sample-total ICC metrics for SV, REL, and CU;
- legacy descriptive agreement summaries, including sample-level 80-percent agreement flags.

## Paradigm-Specific Outputs

If `advanced.cu_paradigms` is empty, DIAAD evaluates the base `sv`/`rel` columns and writes directly under `cu_reliability/`.

If paradigms are configured, DIAAD evaluates each configured paradigm separately. Outputs are written under a paradigm subdirectory and include the paradigm label in the filename.

## Common Problems

If the command skips a paradigm, check that both workbooks contain the required column pair for that paradigm.

If coverage is lower than expected, check whether reliability rows have matching utterance identifiers.

If kappa is missing or surprising, inspect paired values and variance diagnostics. Kappa may be undefined when paired ratings have little or no variability.
