# `transcripts chats` Usage Guide

Use `diaad transcripts chats` when the transcript table is the current source of truth and you need CHAT-style files from that table.

## Before Running

Make sure the transcript table contains the required sheets:

| Sheet | Required content |
|---|---|
| `samples` | One row per sample, including the configured sample identifier column. |
| `utterances` | One row per utterance, including the configured sample identifier column, `speaker`, and `utterance`. |

If present, `position` and `position_sub` are used to sort utterances within each sample.

## Optional Template Header

The command looks for a file matching:

```text
*template_header.cha
```

under the input directory. If one is found, DIAAD uses it as the header template for exported CHAT files. If multiple matching headers exist, the first one is used and a warning is logged. If none exists, DIAAD uses a default CHAT header.

## Output Filenames

DIAAD constructs derived CHAT filenames from sample-level columns. It excludes technical fields such as `input_order`, `shuffled_order`, and `derived_file`. If the available metadata would produce duplicate names, DIAAD appends a row index so the files remain distinct.

For example, a sample row with metadata fields such as site, timepoint, and narrative may produce:

```text
AC25_Pre_BrokenWindow.cha
```

The generated name is also recorded in the output transcript table's `derived_file` column.

## What Is Preserved

The export preserves table-level transcript content: speaker labels, utterance text, utterance order, and comments stored in `%com` rows.

The export also regularizes some punctuation to conservative ASCII forms for CHAT output. This improves compatibility, but it means exported files should be inspected when exact formatting matters.

## Common Problems

If the command cannot find a transcript table, check `advanced.transcript_table_filename` and remove duplicate copies under the active input tree. DIAAD uses exact workbook discovery for this command.

If a sample does not produce a CHAT file, check whether that sample has matching utterance rows under the configured sample identifier column.

If filenames are not informative, inspect the sample-level columns available in the `samples` sheet. The output name is built from those columns after technical columns are excluded.
