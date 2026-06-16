# Transcription-Based Workflow Baseline Quickstart

Most DIAAD discourse-analysis workflows begin with CHAT transcripts and become easier to manage once those transcripts are represented as DIAAD transcript tables. This baseline gives later monologic and dialogic workflows their shared starting structure.

## Baseline Sequence

1. Prepare initial CHAT transcripts outside DIAAD.
2. Review and stabilize those transcripts according to the project protocol.
3. Select a transcription reliability subset:

```bash
diaad transcripts select
```

4. Complete independent reliability transcription outside DIAAD.
5. Evaluate transcription reliability:

```bash
diaad transcripts evaluate
```

6. Reselect only if another reliability round is needed:

```bash
diaad transcripts reselect
```

7. Tabularize the stabilized CHAT transcripts:

```bash
diaad transcripts tabularize
```

8. Inspect the transcript table workbook before generating coding or analysis files.

## Main Output

The key shared artifact is:

```text
transcript_tables/transcript_tables.xlsx
```

It contains sample-level rows, utterance-level rows, metadata diagnostics, stable sample identifiers, stable utterance identifiers, and utterance-order fields.

## Before Branching

Before using the table for Complete Utterances, Word Counting, POWERS, Target Vocabulary Coverage, Templates, or transcript-table-informed Digital Conversational Turns, inspect:

- sample identifiers and metadata;
- speaker labels;
- utterance text;
- `position` and `position_sub`;
- `metadata_mismatch` and the `metadata_mismatches` sheet.

## Read Next

- Transcript tabularization feature: `docs/manual/03_features/01_transcript_tabularization.md`
- Transcripts module: `docs/manual/04_modules/01_transcripts/01_quickstart.md`
- Transcription reliability workflow: `docs/manual/06_workflows/05_transcription_reliability/01_quickstart.md`
- Transcript revision and CHAT export workflow: `docs/manual/06_workflows/06_transcript_table_revision_chat_export/01_quickstart.md`
