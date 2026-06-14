# `transcripts reselect` Quickstart

`diaad transcripts reselect` selects replacement transcription reliability samples after an earlier `transcripts select` round.

## Run

```bash
diaad transcripts reselect --config config
```

## Minimum Inputs

Provide a prior transcription reliability selection workbook under the configured input directory:

```text
transcription_reliability_samples.xlsx
```

A common layout is:

```text
diaad_data/input/
  transcription_reliability_selection/
    transcription_reliability_samples.xlsx
```

## Primary Output

By default, the command writes:

```text
diaad_data/output/diaad_YYMMDD_HHMM/reselected_transcription_reliability/
  reselected_transcription_reliability_samples.xlsx
```

The workbook contains a `reselected_reliability` sheet.

## Immediate Next Step

Inspect the reselected rows and confirm that they exclude the samples selected in the prior round. If the output contains fewer rows than expected, the earlier selection may have exhausted the available candidates.

## Read Next

- [Generated Example I/O](../../../../03_features/04_generated_example_io.md)
- [Transcripts module quickstart](../../01_quickstart.md)
- [Configuration](../../../../02_operation/04_configuration.md)
