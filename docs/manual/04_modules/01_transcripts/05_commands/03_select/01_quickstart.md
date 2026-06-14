# `transcripts select` Quickstart

`diaad transcripts select` selects a subset of transcripts for transcription reliability work and writes a workbook documenting both the selected subset and the full sample frame.

## Run

```bash
diaad transcripts select --config config
```

## Minimum Inputs

The command can use:

- an existing transcript table workbook, preferred when present; or
- CHAT `.cha` files, used as a fallback source for the sample frame.

If CHAT files are available, DIAAD also writes blank reliability `.cha` files containing headers only.

## Important Setting

Set the reliability fraction in `project.yaml`:

```yaml
reliability_fraction: 0.2
```

The value must be greater than `0` and less than or equal to `1`.

## Primary Outputs

By default, the command writes:

```text
diaad_data/output/diaad_YYMMDD_HHMM/transcription_reliability_selection/
  transcription_reliability_samples.xlsx
  *_reliability.cha
```

The workbook contains:

| Sheet | Purpose |
|---|---|
| `reliability_selection` | Rows selected for reliability transcription. |
| `all_transcripts` | Full sample frame with `selected_for_reliability` marked as `0` or `1`. |

## Immediate Next Step

Use the selected rows and blank reliability `.cha` files to organize independent reliability transcription. Do not treat the blank `.cha` files as completed reliability transcripts.

## Read Next

- [Generated Example I/O](../../../../03_features/04_generated_example_io.md)
- [Transcripts module quickstart](../../01_quickstart.md)
- [Configuration](../../../../02_operation/04_configuration.md)
