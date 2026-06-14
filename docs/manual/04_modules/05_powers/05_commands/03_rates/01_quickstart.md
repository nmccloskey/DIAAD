# `powers rates` Quickstart

`diaad powers rates` combines POWERS dialog summaries with speaking-time values to calculate per-minute rates.

## Run

```bash
diaad powers rates --config config
```

## Minimum Inputs

Provide a POWERS analysis workbook and a speaking-time workbook. A common input layout is:

```text
diaad_data/input/
  powers_coding_analysis/
    powers_analysis.xlsx
  speaking_times/
    speaking_times.xlsx
```

The speaking-time workbook should contain the configured sample identifier and speaking-time column. Defaults are:

```text
sample_id
speaking_time
```

## Primary Output

By default, the command writes:

```text
powers_coding_analysis/
  powers_coding_rates.xlsx
```

## Immediate Next Step

Check that `speaking_time` values were entered in seconds and that per-minute columns appear for the expected count-like dialog measures.

## Read Next

- `powers analyze` quickstart: `docs/manual/04_modules/05_powers/05_commands/02_analyze/01_quickstart.md`
- Speaking-time templates: `docs/manual/04_modules/02_templates/05_commands/03_times/01_quickstart.md`
- POWERS module quickstart: `docs/manual/04_modules/05_powers/01_quickstart.md`
- Generated Example I/O: `docs/manual/03_features/04_generated_example_io.md`
