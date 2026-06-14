# `cus rates` Quickstart

`diaad cus rates` combines CU sample summaries with speaking-time values to calculate per-minute CU, SV, and REL rates.

## Run

```bash
diaad cus rates --config config
```

## Minimum Inputs

Provide:

```text
cu_coding_by_sample_long.xlsx
speaking_times.xlsx
```

A common layout is:

```text
diaad_data/input/
  cu_coding_analysis/
    cu_coding_by_sample_long.xlsx
  speaking_times/
    speaking_times.xlsx
```

## Primary Output

By default, the command writes:

```text
cu_coding_analysis/
  cu_coding_rates.xlsx
```

## Immediate Next Step

Inspect missing or zero speaking-time values before interpreting rates. Rates are `NaN` when the denominator is missing or not positive.

## Read Next

- `templates times` usage: `docs/manual/04_modules/02_templates/05_commands/03_times/02_usage_guide.md`
- `cus analyze` usage: `docs/manual/04_modules/03_complete_utterances/05_commands/04_analyze/02_usage_guide.md`
- Configuration: `docs/manual/02_operation/04_configuration.md`
