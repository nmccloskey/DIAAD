# Functional Overview

This page gives a centralized map of DIAAD's current modules, commands, and cross-cutting functions. It is meant as an orientation table, not a substitute for the later command-level pages. The command pages should provide the exact inputs, outputs, configuration settings, and examples needed to run each operation.

## Major Functional Areas

| Functional area | What it does | Main commands or controls | Later detail |
|---|---|---|---|
| Transcript tabularization | Converts CHAT transcripts into stable sample- and utterance-level tables. | `transcripts tabularize` | Transcripts module; transcript-based workflows. |
| Transcript recovery | Reconstructs CHAT-style files from transcript tables when possible. | `transcripts chats` | Transcripts module. |
| Reliability workflows | Selects reliability subsets, evaluates agreement, and creates replacement subsets when needed. | `transcripts select`, `transcripts evaluate`, `transcripts reselect`; module-specific `evaluate` and `reselect` commands | Reliability functionality; module command pages. |
| Coding file generation | Creates workbooks for manual or human-reviewed coding. | `cus files`, `words files`, `powers files`, `turns files`, `templates utterances`, `templates samples`, `templates times` | Module command pages. |
| Coding analysis | Summarizes completed coding workbooks into analysis-ready tables. | `cus analyze`, `words analyze`, `powers analyze`, `turns analyze`, `vocab analyze` | Module command pages. |
| Rate calculation | Combines analysis outputs with speaking-time values to calculate per-minute rates. | `templates times`, `cus rates`, `words rates`, `powers rates`, `vocab rates` | Rate-calculation functionality. |
| Target vocabulary resources | Creates, validates, and applies built-in or custom target vocabulary resources. | `vocab file`, `vocab check`, `vocab analyze` | Target vocabulary module. |
| Blinding | Encodes and decodes configured identifier columns for coding and analysis workflows. | `blinding encode`, `blinding decode`, `advanced.auto_blind`, `advanced.blind_columns` | Blinding module and functionality. |
| Generated examples | Creates synthetic example projects and generated Example I/O documentation. | `diaad examples`, `--for-command`, `--render-docs` | Examples workflow; generated Example I/O view. |
| Configuration and provenance | Resolves defaults, project config, CLI overrides, run paths, logs, manifests, and dry-run outputs. | `--config`, `--set`, `--dry-run-config`; run artifact writing | Operation and configuration pages. |

## Module Summary

| Module | Primary purpose | Typical starting point | Typical output |
|---|---|---|---|
| `examples` | Generate runnable synthetic example projects and generated documentation artifacts. | No study data required. | Example input/output folders and Example I/O markdown. |
| `blinding` | Encode or decode configured identifiers in tabular files. | Files with configured ID columns and, for decoding, a codebook. | Blinded or decoded files plus codebook handling. |
| `transcripts` | Manage CHAT transcript tabularization, transcript reliability, and CHAT reconstruction. | `.cha` transcript files or transcript tables. | Transcript tables, reliability materials, reliability reports, reconstructed CHAT files. |
| `templates` | Create generic coding and timing templates not tied to one specialized analysis module. | Transcript tables or sample-level tables. | Utterance templates, sample templates, speaking-time workbooks, sample subsets. |
| `cus` | Support Complete Utterance coding workflows. | Transcript tables and completed CU coding files. | CU coding workbooks, reliability summaries, analysis workbooks, CU rates. |
| `words` | Support manual word-count coding workflows. | Transcript tables or CU-derived coding inputs. | Word-count workbooks, reliability summaries, word-count analyses, word-count rates. |
| `powers` | Support POWERS coding, selected automation, reliability, analysis, and rates. | Transcript tables and completed POWERS coding files. | POWERS coding workbooks, reliability summaries, analysis workbooks, POWERS rates. |
| `vocab` | Support target vocabulary resource creation, validation, analysis, and rates. | Built-in or custom vocabulary resources plus transcript-derived inputs. | Resource templates/checks, coverage summaries, long-format detail tables, rates. |
| `turns` | Support Digital Conversational Turn coding and analysis. | Transcript tables and completed turn-coding workbooks. | Turn-coding templates, reliability reports, transition and summary analyses. |

## Command Index

### Examples

| Command | Purpose | Primary input | Primary output |
|---|---|---|---|
| `examples` | Generate full or command-specific synthetic example files; optionally regenerate Example I/O docs. | Packaged example specs. | Example project files under the configured output directory; generated markdown when `--render-docs` is used. |

### Blinding

| Command | Purpose | Primary input | Primary output |
|---|---|---|---|
| `blinding encode` | Replace configured identifier columns with blind codes. | Tabular files with configured blinding columns. | Blinded files and codebook artifacts. |
| `blinding decode` | Restore identifiers from a blind codebook when outputs need to be reconnected. | Blinded tabular files plus codebook context. | Decoded files with canonical identifiers restored. |

### Transcripts

| Command | Purpose | Primary input | Primary output |
|---|---|---|---|
| `transcripts select` | Select transcription reliability samples. | CHAT `.cha` files. | Reliability sample lists and reliability transcript templates. |
| `transcripts evaluate` | Evaluate transcription reliability between original and reliability transcripts. | Original/reliability `.cha` pairs. | Agreement metrics and alignment reports. |
| `transcripts reselect` | Select replacement transcription reliability samples after earlier selections have been used. | Original and reliability transcription materials. | New reliability subset materials. |
| `transcripts tabularize` | Convert CHAT transcripts into structured sample and utterance tables. | CHAT `.cha` files. | `transcript_tables.xlsx` by default. |
| `transcripts chats` | Reconstruct CHAT-style files from transcript tables. | Transcript table workbook. | Reconstructed `.cha` files. |

### Templates

| Command | Purpose | Primary input | Primary output |
|---|---|---|---|
| `templates utterances` | Generate generic utterance-level coding templates. | Transcript tables. | Utterance-level coding template workbooks. |
| `templates samples` | Generate generic sample-level or bin-level coding templates. | Transcript tables. | Sample-level coding template workbooks. |
| `templates times` | Generate a speaking-time workbook for later rate calculations. | Transcript tables. | Speaking-time template workbook. |
| `templates subset` | Select a general sample subset from eligible samples. | One Excel workbook with a `samples` sheet and sample identifier column; optional `exclude` column. | Sample subset workbook. |

### Complete Utterances

| Command | Purpose | Primary input | Primary output |
|---|---|---|---|
| `cus files` | Generate Complete Utterance coding and reliability workbooks. | Transcript tables. | CU coding and reliability workbooks. |
| `cus evaluate` | Evaluate Complete Utterance coding reliability. | Completed CU coding and reliability workbooks. | Reliability summaries and reports. |
| `cus reselect` | Create a new CU reliability subset after prior samples have been used. | Completed or existing CU reliability materials. | Replacement CU reliability workbook(s). |
| `cus analyze` | Analyze completed Complete Utterance coding. | Completed CU coding workbooks. | Sample- and utterance-level CU analysis outputs. |
| `cus rates` | Calculate CU rates using speaking-time values. | CU analysis output plus speaking-time workbook. | CU per-minute rate outputs. |

### Word Counting

| Command | Purpose | Primary input | Primary output |
|---|---|---|---|
| `words files` | Generate manual word-count coding and reliability workbooks. | Transcript tables or CU-derived inputs, depending on workflow. | Word-count coding and reliability workbooks. |
| `words evaluate` | Evaluate word-count reliability. | Completed word-count coding and reliability workbooks. | Reliability summaries and agreement reports. |
| `words reselect` | Create a new word-count reliability subset after prior samples have been used. | Completed or existing word-count reliability materials. | Replacement word-count reliability workbook(s). |
| `words analyze` | Analyze completed word-count coding. | Completed word-count workbooks. | Sample-level word-count summaries. |
| `words rates` | Calculate word-count rates using speaking-time values. | Word-count analysis output plus speaking-time workbook. | Word-count per-minute rate outputs. |

### POWERS

| Command | Purpose | Primary input | Primary output |
|---|---|---|---|
| `powers files` | Generate POWERS coding and reliability workbooks, with optional automated first-pass support. | Transcript tables. | POWERS coding and reliability workbooks. |
| `powers analyze` | Analyze completed POWERS coding. | Completed POWERS coding workbooks. | Utterance, turn, speaker, and dialog summaries. |
| `powers rates` | Calculate POWERS rates using speaking-time values. | POWERS analysis output plus speaking-time workbook. | POWERS per-minute rate outputs. |
| `powers evaluate` | Evaluate POWERS reliability. | Completed POWERS coding and reliability workbooks. | Continuous and categorical reliability summaries. |
| `powers reselect` | Create a new POWERS reliability subset after prior samples have been used. | Completed or existing POWERS reliability materials. | Replacement POWERS reliability workbook(s). |

### Target Vocabulary Coverage

| Command | Purpose | Primary input | Primary output |
|---|---|---|---|
| `vocab file` | Generate a blank custom target vocabulary resource template. | No study data required. | JSON resource template. |
| `vocab check` | Validate and summarize active built-in and custom target vocabulary resources. | Built-in resources and optional custom JSON resources. | Validation summary and resource diagnostics. |
| `vocab analyze` | Analyze target vocabulary coverage. | Transcript-derived input tables and active vocabulary resources. | Coverage summaries and long-format detail tables. |
| `vocab rates` | Calculate target vocabulary rates using speaking-time values. | Target vocabulary analysis output plus speaking-time workbook. | Target vocabulary per-minute rate outputs. |

### Digital Conversational Turns

| Command | Purpose | Primary input | Primary output |
|---|---|---|---|
| `turns files` | Generate Digital Conversational Turn coding and reliability templates. | Transcript tables. | Turn-coding and reliability workbooks. |
| `turns evaluate` | Evaluate turn-coding reliability using count and sequence comparisons. | Completed turn-coding and reliability workbooks. | Reliability workbook, reports, and alignment artifacts. |
| `turns reselect` | Create a new turn-coding reliability subset after prior samples have been used. | Existing turn-coding and reliability materials. | Replacement turn reliability template(s). |
| `turns analyze` | Analyze completed Digital Conversational Turn coding. | Completed turn-coding workbooks. | Speaker, group, session, and transition summaries. |

## Common Workflow Shapes

| Workflow | Typical sequence | Notes |
|---|---|---|
| Generated examples | `diaad examples`; optionally `diaad examples --for-command "<command>"` | Best first step for learning expected file layouts. |
| Monologic transcript workflow | `transcripts tabularize` -> `cus files` and/or `words files` -> manual coding -> `cus analyze`, `words analyze`, `vocab analyze` -> rate commands as needed | Most transcript-based workflows begin with transcript tabularization. |
| Clinician-client dialogue workflow | `transcripts tabularize` -> `powers files` -> human review/coding -> `powers analyze` -> `powers rates` | POWERS automation is a first-pass support tool, not a replacement for review. |
| Reliability refresh workflow | module `files` command -> manual reliability coding -> module `evaluate` -> module `reselect` if a replacement subset is needed | Reselection commands are meant for iterative coding projects. |
| Generic coding workflow | `transcripts tabularize` -> `templates utterances` or `templates samples` -> manual coding outside a specialized DIAAD analyzer | Useful when DIAAD should scaffold a project-specific coding task. |
| Speaking-time and rates workflow | `templates times` -> enter speaking times -> run one or more `rates` commands | Speaking-time values are expected in the configured speaking-time column. |
| Blinding workflow | configure `blind_columns` and optionally `auto_blind`; or run `blinding encode` / `blinding decode` explicitly | Blinding supports masking and recovery of identifiers but does not guarantee de-identification. |

## Dependency Notes

| Dependency type | Commands affected | Behavior |
|---|---|---|
| CHAT files | `transcripts tabularize`, `transcripts select` | These commands read `.cha` transcripts directly. |
| Transcript tables | `cus files`, `vocab analyze`, `powers files`, `transcripts chats`, `templates utterances`, `templates samples`, `templates times`, `turns files`; fallback input for `words files` | These commands need transcript tables. If `auto_tabularize` is false, run `transcripts tabularize` first. |
| CU coding files | Preferred input for `words files` | Word-count file generation uses CU coding files when present so neutral/non-countable utterances can be excluded before first-pass counts. |
| Completed coding files | `cus evaluate`, `cus analyze`, `words evaluate`, `words analyze`, `powers evaluate`, `powers analyze`, `turns evaluate`, `turns analyze` | DIAAD expects human-entered or human-reviewed coding files in the configured input location. |
| Speaking-time tables | `cus rates`, `words rates`, `powers rates`, `vocab rates` | Use `templates times` to create a starting workbook. |
| spaCy model | `powers files` when automation is enabled | Install `diaad[nlp]` and the configured spaCy model, or disable automation if appropriate. |
| Target vocabulary resources | `vocab check`, `vocab analyze` | Built-in resources are always available; custom resources can be configured with `target_vocabulary_resource_path`. |
