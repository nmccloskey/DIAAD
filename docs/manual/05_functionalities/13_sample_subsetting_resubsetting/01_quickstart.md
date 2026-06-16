# General Sample Subsetting and Re-Subsetting Quickstart

`templates subset` is DIAAD's general sample-selection utility. It can support reliability planning, pilot sample selection, quality-control review, or any protocol that needs a random subset from a workbook with a `samples` sheet.

## Basic Command

```bash
diaad templates subset
```

The input directory must contain exactly one `.xlsx` workbook. That workbook must include:

```text
samples
```

The `samples` sheet must contain the configured sample identifier column, usually:

```text
sample_id
```

## Output

The command writes:

```text
coding_templates/sample_subset.xlsx
```

The workbook contains:

- `samples`: all samples with `selected` and `excluded` status;
- `subset`: only selected samples.

## Re-Subsetting

If the input `samples` sheet has an `exclude` column with `0` and `1` values, DIAAD samples only from rows whose collapsed sample-level exclusion value is `0`.

Use this when a later selection should avoid samples already used in a prior round.

## Read Next

- Templates module: `docs/manual/04_modules/02_templates/`
- `templates subset` command: `docs/manual/04_modules/02_templates/05_commands/04_subset/01_quickstart.md`
- Reliability functionality: `docs/manual/05_functionalities/12_reliability_selection_evaluation_reselection/01_quickstart.md`
