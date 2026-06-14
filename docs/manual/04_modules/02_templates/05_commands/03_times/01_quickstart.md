# `templates times` Quickstart

`diaad templates times` creates a blank speaking-time workbook keyed by sample identifier. Later rate commands use completed speaking-time values as denominators.

## Run

```bash
diaad templates times --config config
```

## Minimum Inputs

Provide one configured transcript table workbook in the input directory or current run output directory. By default, DIAAD looks for:

```text
transcript_tables.xlsx
```

## Primary Output

By default, the command writes:

```text
coding_templates/
  speaking_times.xlsx
```

The workbook uses a `coding_template` sheet with:

```text
sample_id
speaking_time
```

## Immediate Next Step

Enter speaking-time values in seconds. DIAAD's rate utilities convert `speaking_time` to `speaking_minutes` by dividing by 60.

## Read Next

- Templates module quickstart: `docs/manual/04_modules/02_templates/01_quickstart.md`
- Configuration: `docs/manual/02_operation/04_configuration.md`
- Generated Example I/O: `docs/manual/03_features/04_generated_example_io.md`
