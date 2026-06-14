# `templates times` Usage Guide

Use `diaad templates times` when later rate calculations need per-sample speaking-time values.

## Before Running

Create or provide transcript tables first. The command reads the transcript table `samples` sheet and requires the configured sample identifier column.

The output contains one row per unique sample identifier.

## Entering Values

Enter speaking-time values in seconds in the `speaking_time` column.

DIAAD rate utilities later standardize this table by:

1. reading `sample_id` and the configured speaking-time column;
2. coercing values to numeric;
3. summing duplicate sample IDs if they are present;
4. calculating `speaking_minutes = speaking_time / 60`.

Because of that downstream behavior, entering minutes instead of seconds will inflate per-minute rates.

## Filename Alignment

The template command currently writes:

```text
coding_templates/speaking_times.xlsx
```

The shared rate utilities default to the same filename. If a project changes `advanced.speaking_time_filename` for later rate commands, keep the completed workbook name and the rate configuration aligned.

## Common Problems

If a sample is missing, check the transcript table `samples` sheet and the configured sample identifier column.

If a rate command later reports missing speaking time, check that the completed workbook is in the active input tree for that rate command and that the filename matches `advanced.speaking_time_filename`.

If values are nonnumeric, rate commands retain those rows with missing standardized speaking-time values.
