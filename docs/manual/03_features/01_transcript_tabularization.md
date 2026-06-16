# Transcript Tabularization

Transcript tabularization is the usual first step for DIAAD's transcript-based workflows. It converts CHAT `.cha` files into an Excel workbook that separates sample-level metadata from utterance-level transcript content, while preserving stable identifiers that later coding, reliability, blinding, and analysis steps can share.

This page is a conceptual orientation. Later module and command pages describe the exact command syntax, input layout, and output files for `diaad transcripts tabularize` and related transcript commands.

## What Tabularization Creates

By default, `transcripts tabularize` writes:

```text
diaad_data/output/diaad_YYMMDD_HHMM/transcript_tables/transcript_tables.xlsx
```

The workbook contains three sheets:

| Sheet | Purpose |
|---|---|
| `samples` | One row per transcript/sample, including the sample identifier, original file information, ordering fields, configured metadata, and a metadata-mismatch flag. |
| `utterances` | One row per utterance, including the sample identifier, utterance identifier, position fields, speaker label, utterance text, and comments. |
| `metadata_mismatches` | Diagnostic rows for configured metadata fields that could not be resolved cleanly from the source files. |

The default transcript table filename is controlled by `advanced.transcript_table_filename`, which defaults to `transcript_tables.xlsx`.

## The Relational Scaffold

DIAAD uses transcript tables as an Excel-centered relational scaffold rather than a database server. The important idea is that the same identifiers can appear across many files:

- `sample_id` identifies the transcript/sample.
- `utterance_id` identifies an utterance within a sample.
- together, sample and utterance identifiers let later files reconnect coding decisions, reliability rows, blinding codebooks, analysis summaries, and rate calculations.

The identifier column names are configurable through `advanced.sample_id_column` and `advanced.utterance_id_column`. The default names are `sample_id` and `utterance_id`.

This structure is especially useful because discourse workflows often move between transcript text, metadata, human coding, automated first passes, reliability subsets, and aggregate analysis. A stable table gives those steps a shared coordinate system.

## Metadata and File Context

The `samples` sheet stores source file context such as file name, extension, directory, and input ordering. Projects may also configure `metadata_fields` so DIAAD can extract study-specific metadata from file names, folder paths, or transcript content conventions.

Metadata extraction is project-specific. When a field cannot be resolved cleanly, DIAAD records the issue in the transcript-table output instead of silently treating the metadata as certain. This makes the table useful both as an analysis input and as an early data-quality check.

## Revision Tolerance

Coding workflows often reveal transcript errors. DIAAD transcript tables are designed to tolerate careful edits:

- update the `utterance` text when a transcript row needs correction;
- preserve the sample and utterance identifiers when the row represents the same analytic unit;
- use `position` and `position_sub` to keep utterance ordering explicit, including inserted rows;
- keep metadata and identifier columns stable unless there is a deliberate reason to revise them.

The inverse command, `transcripts chats`, exports CHAT-style files from transcript tables. It is best understood as a revision-export or detabularization path, not as a promise that every source file can be perfectly round-tripped with every original formatting detail. It writes reconstructed `.cha` files under `chat_files/` and writes an updated transcript table copy that records the derived file names.

## Where Transcript Tables Are Required

Many DIAAD commands require transcript tables, including commands that create Complete Utterance, POWERS, target vocabulary, generic coding-template, and speaking-time files from transcript content. Digital Conversational Turns can also use transcript tables as a fallback source for speaker-sequence analysis when no primary DCT workbook is found.

If a later command needs transcript tables and `transcripts tabularize` is not part of the same run, DIAAD searches the input directory and current run output directory for the configured transcript table filename. If it cannot find a usable table, the command fails unless `project.auto_tabularize` is set to `true`.

`auto_tabularize` defaults to `false`. This keeps the usual workflow explicit: create transcript tables first, inspect them, and then use them downstream.

## Important Exceptions

Transcript tabularization is central, but it is not universal.

Some DIAAD workflows may begin without transcript tables, such as:

- transcription reliability selection and evaluation from CHAT files;
- Digital Conversational Turn workflows that are used before or instead of transcription;
- general sample subsetting workflows based on an existing workbook;
- workflows that begin from already-created DIAAD transcript tables;
- future or project-specific workflows built from external tables with stable identifiers.

The broader principle is not "everything must start as CHAT." The broader principle is that reproducible workflows need stable identifiers and predictable tables. Transcript tabularization is DIAAD's main way to create that structure for transcript-based discourse analysis.

## Read Next

- Configuration: `docs/manual/02_operation/04_configuration.md`
- Functional overview: `docs/manual/01_overview/04_functional_overview.md`
- Exact file name matching: `docs/manual/03_features/03_exact_file_name_matching.md`
- Generated Example I/O: `docs/manual/03_features/04_generated_example_io.md`

## Draft Review Notes

Before publication, review the relational-database framing, the revision-export wording for `transcripts chats`, and the guidance for editing `position` and `position_sub` against the intended user-facing revision workflow.
