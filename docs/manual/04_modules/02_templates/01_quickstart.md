# Templates Module Quickstart

The Templates module creates generic workbooks for human coding, speaking-time entry, and sample subsetting. It is supportive infrastructure: it helps projects create organized files even when the coding paradigm is not one of DIAAD's specialized analyzers.

## Commands

| Command | Main use |
|---|---|
| `diaad templates utterances` | Create an utterance-level coding workbook and reliability workbook. |
| `diaad templates samples` | Create a sample-level or bin-level coding workbook and reliability workbook. |
| `diaad templates times` | Create a speaking-time workbook for later rate calculations. |
| `diaad templates subset` | Select a general sample subset from an input workbook. |

## Typical Uses

Use Templates when you need:

- a blank utterance-level workbook for project-specific manual coding;
- a sample-level or binned template for study-specific ratings;
- a speaking-time workbook to support `rates` commands;
- a random subset for reliability, piloting, or another protocol-driven purpose.

Transcript-table-based template commands usually expect `transcript_tables.xlsx` or the configured transcript table filename to be available.

## Common Outputs

| Command | Typical outputs |
|---|---|
| `templates utterances` | `coding_templates/utterance_coding_template.xlsx`, reliability template, codebook |
| `templates samples` | `coding_templates/sample_coding_template.xlsx`, reliability template, codebook |
| `templates times` | `coding_templates/speaking_times.xlsx` |
| `templates subset` | `coding_templates/sample_subset.xlsx` |

## Read Next

- [Transcript tabularization](../../03_features/01_transcript_tabularization.md)
- [Exact file name matching](../../03_features/03_exact_file_name_matching.md)
- [Configuration](../../02_operation/04_configuration.md)

Later command pages describe the exact columns, reliability subset behavior, and expected input workbook structure for each template command.
