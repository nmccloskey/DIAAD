# General Sample Subsetting and Re-Subsetting Usage Guide

Use general subsetting when a project needs an auditable random sample but the workflow is not fully covered by a module-specific selector.

## When To Use It

Good uses include:

- selecting a pilot set for protocol refinement;
- selecting samples for a custom coding workflow;
- selecting quality-control review material;
- selecting reliability material for a local paradigm not directly supported by DIAAD;
- selecting replacement material while excluding samples already used.

If a dedicated DIAAD command exists for the workflow, use that command first. For example, transcription reliability should usually use `transcripts select`, and module-generated coding files should use their module's reliability workbook support.

## Input Workbook

The input directory should contain exactly one `.xlsx` workbook. Put unrelated workbooks elsewhere before running.

The workbook must have a sheet named:

```text
samples
```

That sheet must contain the configured sample identifier column. By default:

```text
sample_id
```

Rows with blank sample identifiers are rejected.

## Selection Fraction

The command uses:

```yaml
project:
  reliability_fraction: 0.2
```

Although the setting name says `reliability_fraction`, the command is general. The selected count is the ceiling of the fraction times the number of samples, with a minimum of one sample.

## Excluding Prior Samples

To avoid selecting previously used samples, add an `exclude` column to the `samples` sheet.

Allowed values are:

```text
0
1
```

If a sample appears more than once and any row is marked `1`, DIAAD treats the whole sample as excluded. The target subset size is still calculated from the full sample set, then capped by the number of eligible candidates.

## Output Review

Review both sheets in `sample_subset.xlsx`:

- `samples` documents every sample and whether it was selected or excluded;
- `subset` contains only selected samples.

The command does not create coding files or analysis outputs. It creates a selection artifact that a project can use in a later protocol.

## Read Next

- Exact file name matching: `docs/manual/03_features/03_exact_file_name_matching.md`
- Configuration: `docs/manual/02_operation/04_configuration.md`
- Reliability usage guide: `docs/manual/05_functionalities/12_reliability_selection_evaluation_reselection/02_usage_guide.md`
