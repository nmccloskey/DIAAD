# `transcripts tabularize` Quickstart

`diaad transcripts tabularize` converts CHAT `.cha` transcripts into DIAAD transcript tables. It is the usual first command for transcript-based DIAAD workflows.

## Run

```bash
diaad transcripts tabularize --config config
```

The `--config config` argument is optional when the project has a `config/` directory in the working directory. See [Command-line operation](../../../../02_operation/02_command_line.md) for shared CLI behavior.

## Minimum Inputs

Place CHAT files under the configured input directory, which defaults to:

```text
diaad_data/input/
```

DIAAD searches recursively for `.cha` files. During ordinary transcript loading it excludes directories named by `advanced.reliability_dirname`, which defaults to `reliability`.

## Primary Output

By default, the command writes:

```text
diaad_data/output/diaad_YYMMDD_HHMM/transcript_tables/transcript_tables.xlsx
```

The workbook contains:

| Sheet | What to check |
|---|---|
| `samples` | One row per transcript/sample, including source file fields, ordering fields, configured metadata, and `metadata_mismatch`. |
| `utterances` | One row per utterance, including `sample_id`, `utterance_id`, position fields, speaker, utterance text, and comment. |
| `metadata_mismatches` | Diagnostics for metadata fields that did not resolve cleanly. |

## Immediate Next Step

Open the transcript table and inspect it before generating downstream coding files. In particular, check `metadata_mismatch`, speaker labels, utterance segmentation, and any project-specific metadata fields.

## Read Next

- [Transcript tabularization feature](../../../../03_features/01_transcript_tabularization.md)
- [Transcripts module quickstart](../../01_quickstart.md)
- [Configuration](../../../../02_operation/04_configuration.md)
- [Generated Example I/O](../../../../03_features/04_generated_example_io.md)
