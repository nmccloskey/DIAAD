# Transcripts Module Quickstart

The Transcripts module manages the transcript-facing parts of DIAAD: reading CHAT files, creating transcript tables, selecting and evaluating transcription reliability samples, reselecting reliability samples when needed, and exporting transcript tables back to CHAT-style files.

Most transcript-based DIAAD workflows begin here.

## Commands

| Command | Main use |
|---|---|
| `diaad transcripts tabularize` | Convert CHAT `.cha` files into transcript tables. |
| `diaad transcripts chats` | Export CHAT-style files from transcript tables, usually after table-based revision. |
| `diaad transcripts select` | Select a transcription reliability subset. |
| `diaad transcripts evaluate` | Compare original and reliability transcripts. |
| `diaad transcripts reselect` | Select replacement reliability samples after a prior reliability round. |

## Typical Start

For a transcript-based project, place CHAT files under the configured input directory and run:

```bash
diaad transcripts tabularize --config config
```

By default, DIAAD writes:

```text
diaad_data/output/diaad_YYMMDD_HHMM/transcript_tables/transcript_tables.xlsx
```

That workbook becomes the common input for many later modules, including Complete Utterances, Word Counting, POWERS, Target Vocabulary Coverage, Templates, and some Digital Conversational Turn file-generation workflows.

## What To Inspect

After tabularization, inspect the transcript table workbook before continuing. In particular, check:

- `samples` for sample IDs, source files, ordering fields, metadata, and metadata mismatch flags;
- `utterances` for speaker labels, utterance text, comments, `position`, and `position_sub`;
- `metadata_mismatches` for extraction problems that need correction before downstream analysis.

## Common Paths

| Task | Output location |
|---|---|
| Transcript tables | `transcript_tables/transcript_tables.xlsx` |
| CHAT export | `chat_files/*.cha` |
| Transcription reliability selection | `transcription_reliability_selection/transcription_reliability_samples.xlsx` |
| Transcription reliability evaluation | `transcription_reliability_evaluation/` |
| Transcription reliability reselection | `reselected_transcription_reliability/` |

## Read Next

- [Transcript tabularization feature](../../03_features/01_transcript_tabularization.md)
- [Exact file name matching](../../03_features/03_exact_file_name_matching.md)
- [Configuration](../../02_operation/04_configuration.md)
- [Functional overview](../../01_overview/03_functional_overview.md)

Later command pages describe exact input layouts, workbook sheets, and command-specific options.
