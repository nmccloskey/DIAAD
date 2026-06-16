# Transcript Preprocessing, Tabularization, Auto-Tabularization, and CHAT Export Quickstart

DIAAD transcript workflows usually begin by converting CHAT `.cha` files into transcript tables. The transcript table becomes the shared workbook that later coding, reliability, blinding, analysis, and rate steps can read.

## Safest Starting Point

For most transcript-based projects, run tabularization as an explicit first step:

```bash
diaad transcripts tabularize
```

Then inspect the generated workbook before using it downstream:

```text
diaad_data/output/diaad_YYMMDD_HHMM/transcript_tables/transcript_tables.xlsx
```

The workbook has separate sheets for sample-level rows, utterance-level rows, and metadata mismatch diagnostics.

## Auto-Tabularization

`project.auto_tabularize` defaults to `false`. Leave it that way for ordinary projects.

When it is `false`, commands that require transcript tables stop if DIAAD cannot find the configured transcript table workbook. This is intentional: it gives you a chance to tabularize once, review the table, and treat the reviewed workbook as the canonical source for later steps.

Set `auto_tabularize: true` only when you deliberately want a downstream command to create transcript tables from available `.cha` files during the same run.

## CHAT Export

Use `transcripts chats` when you need CHAT-style files reconstructed from an edited transcript table:

```bash
diaad transcripts chats
```

This is best understood as a revision-export path, not as a requirement that every project round-trip transcript tables back to CHAT.

## Read Next

- Transcript tabularization feature: `docs/manual/03_features/01_transcript_tabularization.md`
- Transcripts module quickstart: `docs/manual/04_modules/01_transcripts/01_quickstart.md`
- Configuration: `docs/manual/02_operation/04_configuration.md`
- Revision handling: `docs/manual/05_functionalities/11_revision_handling/01_quickstart.md`
