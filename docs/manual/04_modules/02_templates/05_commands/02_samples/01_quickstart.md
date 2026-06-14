# `templates samples` Quickstart

`diaad templates samples` creates blank sample-level coding workbooks from a DIAAD transcript table. It can expand samples into bins and create a reliability workbook for a custom sample-level coding protocol.

## Run

```bash
diaad templates samples --config config
```

## Minimum Inputs

Provide one configured transcript table workbook in the input directory or current run output directory. By default, DIAAD looks for:

```text
transcript_tables.xlsx
```

## Primary Outputs

By default, the command writes:

```text
coding_templates/
  sample_coding_template.xlsx
  sample_reliability_template.xlsx
  sample_template_codebook.xlsx
```

The codebook is written only when configured blinding produces one.

Each template workbook uses a `coding_template` sheet. Typical columns include `sample_id`, `coder_id`, optional `stimulus`, and `bin`.

## Immediate Next Step

Inspect the bins, coder assignments, and reliability rows before the workbook is used for coding. Add project-specific coding columns only after confirming the generated scaffold is correct.

## Read Next

- Templates module quickstart: `docs/manual/04_modules/02_templates/01_quickstart.md`
- Transcript tabularization: `docs/manual/03_features/01_transcript_tabularization.md`
- Configuration: `docs/manual/02_operation/04_configuration.md`
- Generated Example I/O: `docs/manual/03_features/04_generated_example_io.md`
