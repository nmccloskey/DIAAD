# `turns files` Quickstart

`diaad turns files` creates blank Digital Conversational Turns coding workbooks.

## Run

```bash
diaad turns files --config config
```

## Minimum Inputs

The command needs transcript tables so it can identify the sample rows to scaffold. In a typical project, DIAAD uses:

```text
transcript_tables.xlsx
```

The current CLI dispatch checks for transcript tables before `turns files` runs, and may create them through the transcript-table prerequisite path when the project inputs support that.

## Primary Outputs

By default, the command writes:

```text
coding_templates/
  conversation_turns_template.xlsx
  conversation_turns_reliability_template.xlsx
```

When configured blinding is active for coding, it also writes:

```text
coding_templates/
  conversation_turns_template_codebook.xlsx
```

## Coding Task

Coders fill the `turns` column with compact digit strings such as:

```text
0.1..23.0.12
```

Each digit represents one speaker turn. In the bundled convention, `0` is the clinician or other non-client interlocutor category, and `1` through `9` identify client or participant speakers.

## Immediate Next Step

Open the primary and reliability templates, confirm that the sample, coder, session, and bin rows are expected, and then fill `turns` with digit strings according to the project's DCT coding protocol.

## Read Next

- `turns files` usage guide: `docs/manual/04_modules/07_digital_conversational_turns/05_commands/01_files/02_usage_guide.md`
- Digital Conversational Turns quickstart: `docs/manual/04_modules/07_digital_conversational_turns/01_quickstart.md`
- Transcript tabularization: `docs/manual/03_features/01_transcript_tabularization.md`
- Configuration: `docs/manual/02_operation/04_configuration.md`
- Generated Example I/O: `docs/manual/03_features/04_generated_example_io.md`
