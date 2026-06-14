# `powers reselect` Quickstart

`diaad powers reselect` creates replacement POWERS reliability material after an earlier reliability workbook has already been used.

## Run

```bash
diaad powers reselect --config config
```

## Minimum Inputs

Provide prior POWERS primary and reliability workbooks:

```text
powers_coding.xlsx
powers_reliability_coding.xlsx
```

A common input layout is:

```text
diaad_data/input/
  powers_coding/
    powers_coding.xlsx
    powers_reliability_coding.xlsx
```

## Primary Output

By default, the command writes:

```text
reselected_powers_reliability/
  reselected_powers_reliability_coding.xlsx
```

## Immediate Next Step

Review the selected samples, coder IDs, and any automated first-pass values before distributing the reselected reliability workbook.

## Read Next

- `powers files` usage guide: `docs/manual/04_modules/05_powers/05_commands/01_files/02_usage_guide.md`
- `powers evaluate` quickstart: `docs/manual/04_modules/05_powers/05_commands/04_evaluate/01_quickstart.md`
- Configuration: `docs/manual/02_operation/04_configuration.md`
- Generated Example I/O: `docs/manual/03_features/04_generated_example_io.md`
