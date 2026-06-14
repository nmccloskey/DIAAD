# `powers files` Quickstart

`diaad powers files` creates POWERS coding and reliability workbooks from DIAAD transcript tables.

## Run

```bash
diaad powers files --config config
```

## Minimum Input

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
powers_coding/
  powers_coding.xlsx
  powers_reliability_coding.xlsx
  powers_blind_codebook.xlsx
```

The primary workbook includes an `utterance_coding` sheet and a `section_e` sheet. The blind codebook is written only when configured blinding is active.

## Automated First-Pass Support

When `project.automate_powers` is true and the configured spaCy model is available, DIAAD prepopulates:

```text
speech_units
filled_pauses
content_words
num_nouns
tagged_utterance
```

Other POWERS coding fields remain for human coding or review.

## Immediate Next Step

Open `powers_coding.xlsx` and confirm that identifiers, coder assignments, automated first-pass values, blank manual coding fields, and the `section_e` sheet match the intended protocol before distributing the workbook.

## Read Next

- POWERS module quickstart: `docs/manual/04_modules/05_powers/01_quickstart.md`
- POWERS research context: `docs/manual/04_modules/05_powers/03_research_context.md`
- Installation: `docs/manual/02_operation/01_installation.md`
- Configuration: `docs/manual/02_operation/04_configuration.md`
- Generated Example I/O: `docs/manual/03_features/04_generated_example_io.md`
