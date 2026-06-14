# `templates utterances` Quickstart

`diaad templates utterances` creates blank utterance-level coding workbooks from a DIAAD transcript table. It is useful when a project needs a structured manual-coding file for a custom utterance-level protocol.

## Run

```bash
diaad templates utterances --config config
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

By default, the command writes under the timestamped run directory:

```text
coding_templates/
  utterance_coding_template.xlsx
  utterance_reliability_template.xlsx
  utterance_template_codebook.xlsx
```

The codebook is written only when configured blinding produces one.

Each template workbook uses a `coding_template` sheet. Typical columns include `sample_id`, `utterance_id`, `coder_id`, optional `stimulus`, and `utterance`.

## Immediate Next Step

Open the primary and reliability workbooks before distributing them. Confirm that coder assignments, identifiers, stimulus labels, and utterance text match the intended coding protocol.

## Read Next

- Templates module quickstart: `docs/manual/04_modules/02_templates/01_quickstart.md`
- Transcript tabularization: `docs/manual/03_features/01_transcript_tabularization.md`
- Configuration: `docs/manual/02_operation/04_configuration.md`
- Generated Example I/O: `docs/manual/03_features/04_generated_example_io.md`
