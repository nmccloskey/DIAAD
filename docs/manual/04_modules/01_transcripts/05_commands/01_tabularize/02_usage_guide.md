# `transcripts tabularize` Usage Guide

Use `diaad transcripts tabularize` when a project has CHAT transcripts and later DIAAD commands need table-based sample and utterance data.

## Before Running

Prepare the input directory with one or more `.cha` files. A common layout is:

```text
your_project/
  config/
    project.yaml
    advanced.yaml
  diaad_data/
    input/
      chat/
        sample_001.cha
        sample_002.cha
```

Configure metadata fields before tabularization if the transcript table should contain study-specific columns:

```yaml
metadata_fields:
  participant_id: P\d+
  timepoint:
    - pre
    - post
```

Metadata fields are project-specific. If extraction is ambiguous or fails, DIAAD records the issue in the workbook instead of silently treating the value as certain.

## Important Settings

| Setting | Default | Effect |
|---|---|---|
| `project.input_dir` | `diaad_data/input` | Root directory where CHAT files are discovered. |
| `project.output_dir` | `diaad_data/output` | Base directory for timestamped run outputs. |
| `project.random_seed` | `99` | Seed used for shuffled sample order. |
| `project.shuffle_samples` | `true` | Whether sample IDs are assigned from shuffled file order. |
| `project.metadata_fields` | `{}` | Metadata columns to extract into the `samples` sheet. |
| `advanced.transcript_table_filename` | `transcript_tables.xlsx` | Output workbook filename. |
| `advanced.sample_id_column` | `sample_id` | Sample identifier column name. |
| `advanced.utterance_id_column` | `utterance_id` | Utterance identifier column name. |

Use `--dry-run-config` when changing settings:

```bash
diaad transcripts tabularize --config config --dry-run-config --dry-run-config-format yaml
```

## Output Workbook

The `samples` sheet contains source-file fields, ordering fields, configured metadata, and `metadata_mismatch`. The `utterances` sheet contains utterance-level transcript content and identifiers. The `metadata_mismatches` sheet contains one diagnostic row for each configured metadata field that could not be resolved cleanly.

Default identifier values look like:

| Identifier | Example | Scope |
|---|---|---|
| `sample_id` | `S001` | One per transcript/sample. |
| `utterance_id` | `U0001` | Resets within each sample. |

The padding expands when needed, so larger projects keep sortable identifiers.

## Editing After Tabularization

Transcript tables are intended to be inspectable and revision-tolerant. When revising:

- preserve `sample_id` and `utterance_id` when the analytic unit is unchanged;
- correct `utterance` text when transcript errors are found;
- use `position` and `position_sub` to keep ordering explicit, especially for inserted rows;
- avoid changing configured metadata fields unless the change is a documented correction.

After table-based revision, use `diaad transcripts chats` only when you need exported CHAT-style files from the revised table.

## Common Problems

If no `.cha` files are found, check `project.input_dir` and the directory from which the command was run.

If metadata fields are unexpectedly blank, inspect the `metadata_mismatches` sheet and compare the configured metadata patterns with the source file paths.

If later commands cannot find the workbook, preserve the configured transcript table filename or update `advanced.transcript_table_filename` consistently. See Exact file name matching (`docs/manual/03_features/03_exact_file_name_matching.md`).
