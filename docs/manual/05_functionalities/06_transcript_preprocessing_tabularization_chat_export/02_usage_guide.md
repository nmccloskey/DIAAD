# Transcript Preprocessing, Tabularization, Auto-Tabularization, and CHAT Export Usage Guide

This functionality covers the shared transcript preparation path that many DIAAD modules rely on. Command-specific pages explain individual commands; this page explains the common workflow decisions.

## Recommended Workflow

Start with CHAT files in the configured input directory:

```text
diaad_data/input/
  sample_001.cha
  sample_002.cha
```

Run:

```bash
diaad transcripts tabularize
```

Review the output workbook:

```text
transcript_tables/
  transcript_tables.xlsx
```

Inspect at least:

- the `samples` sheet, especially file names, sample identifiers, metadata fields, and `metadata_mismatch`;
- the `utterances` sheet, especially speaker labels, utterance text, comments, and row order;
- the `metadata_mismatches` sheet, if any configured metadata field did not resolve cleanly.

After review, use that workbook as the input for transcript-derived modules such as Complete Utterances, Word Counting, POWERS, Target Vocabulary Coverage, Templates, and some Digital Conversational Turns support.

## Moving Transcript Tables Forward

Each DIAAD run writes to a timestamped output directory. A later run does not automatically search every prior output directory. If you want a later command to use a reviewed transcript table, place the workbook where DIAAD will search for it, usually under the configured input directory while preserving the configured filename:

```text
diaad_data/input/transcript_tables/transcript_tables.xlsx
```

The exact transcript table filename is controlled by `advanced.transcript_table_filename`.

## When Auto-Tabularization Helps

`auto_tabularize` can be useful for quick exploratory runs or generated examples. If a command requires transcript tables and none are found in the input directory or current run output directory, DIAAD can create them from available `.cha` files when `project.auto_tabularize: true`.

For research data, this convenience can create avoidable ambiguity:

- a downstream run may silently create a fresh table instead of using a reviewed table;
- sample IDs may be regenerated from the current file set and shuffle settings;
- multiple transcript table copies may appear across timestamped outputs.

The safer pattern is to tabularize, inspect, then run downstream commands against the reviewed table.

## CHAT Export From Tables

`transcripts chats` loads the configured transcript table, reconstructs CHAT-style files, and writes them under:

```text
chat_files/
```

It also writes an updated transcript table copy that records `derived_file` names. If a file matching `*template_header.cha` is present, DIAAD uses it as the exported CHAT header template; otherwise it writes a default header.

Use this when:

- you corrected transcript-table text and need a CHAT-style export;
- you want a portable set of revised transcript files;
- you need a downstream tool that expects `.cha` files.

Do not treat CHAT export as a lossless preservation tool for every original formatting detail. DIAAD's transcript table is the main analysis scaffold; exported CHAT files are derived artifacts.

## Read Next

- Transcripts commands: `docs/manual/04_modules/01_transcripts/05_commands/`
- Exact file name matching: `docs/manual/03_features/03_exact_file_name_matching.md`
- Metadata extraction: `docs/manual/05_functionalities/08_metadata_extraction/02_usage_guide.md`
- Configurable identifiers: `docs/manual/05_functionalities/10_configurable_sample_utterance_identifiers/02_usage_guide.md`
