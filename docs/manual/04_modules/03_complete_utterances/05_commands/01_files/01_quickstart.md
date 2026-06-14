# `cus files` Quickstart

`diaad cus files` creates Complete Utterance coding and reliability workbooks from DIAAD transcript tables.

## Run

```bash
diaad cus files --config config
```

## Minimum Inputs

Provide one configured transcript table workbook in the input directory or current run output directory. By default, DIAAD looks for:

```text
transcript_tables.xlsx
```

A common input layout is:

```text
diaad_data/input/
  transcript_tables/
    transcript_tables.xlsx
```

## Primary Outputs

By default, the command writes:

```text
cu_coding/
  cu_coding.xlsx
  cu_reliability_coding.xlsx
  cu_blind_codebook.xlsx
```

The reliability workbook is produced from the configured reliability fraction. The blind codebook is written only when configured blinding is active.

## Immediate Next Step

Open `cu_coding.xlsx` and confirm that sample identifiers, utterance identifiers, speaker labels, coder assignment columns, and CU coding columns match the intended protocol before distributing the workbook for coding.

## Read Next

- Complete Utterances module quickstart: `docs/manual/04_modules/03_complete_utterances/01_quickstart.md`
- Transcript tabularization: `docs/manual/03_features/01_transcript_tabularization.md`
- Configuration: `docs/manual/02_operation/04_configuration.md`
- Generated Example I/O: `docs/manual/03_features/04_generated_example_io.md`
