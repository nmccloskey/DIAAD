# `cus analyze` Usage Guide

Use `diaad cus analyze` after CU coding has been completed and checked.

## Required Workbook

DIAAD searches the input directory and current run output directory for exactly one:

```text
cu_coding.xlsx
```

The workbook must contain the configured sample identifier column and at least one valid SV/REL pair.

Supported coding columns include:

```text
sv
rel
sv_<PARADIGM>
rel_<PARADIGM>
c2_sv
c2_rel
c2_sv_<PARADIGM>
c2_rel_<PARADIGM>
```

Additional prefixed coder columns following the `cN_sv` and `cN_rel` pattern can also be detected.

Coder identifier columns (`coder_id`) are not required.

## CU Derivation

For each detected SV/REL pair, DIAAD derives CU as:

| SV | REL | Derived CU |
|---|---|---|
| `1` | `1` | `1` |
| both present, not both `1` | | `0` |
| both missing | | missing |
| one missing and one present | | missing, with an inconsistency logged |

Rows whose speaker label appears in `project.exclude_speakers` are dropped before summary when a `speaker` column is present.

## Outputs

`cu_coding_by_utterance.xlsx` contains utterance-level coding with derived CU columns added.

`cu_coding_by_sample_long.xlsx` contains one row per sample, coder, and paradigm combination. It includes counts, percentages, missingness, and inconsistency counts.

`cu_coding_by_sample.xlsx` contains a wide sample-level summary with one set of columns per coder/paradigm combination.

## Blinding Behavior

If a coding-stage blind codebook is available, the analysis path can reconnect sample identifiers before writing outputs. If analysis-stage blinding is configured, DIAAD can then write blinded analysis outputs plus `cu_analysis_blind_codebook.xlsx` and `cu_analysis_blinding_diagnostics.xlsx`.

## Common Problems

If no outputs appear, check that the workbook contains a valid SV/REL pair.

If expected paradigm columns are skipped, check `advanced.cu_paradigms`. When configured, it filters which paradigms are analyzed.

If `sv_rel_inconsistent` is greater than zero, inspect rows where exactly one of SV or REL was left blank.
