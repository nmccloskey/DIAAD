# `transcripts chats` Quickstart

`diaad transcripts chats` exports CHAT-style `.cha` files from a DIAAD transcript table. Use it when a table has been revised or prepared and you need transcript files reconstructed from the table rows.

## Run

```bash
diaad transcripts chats --config config
```

## Minimum Inputs

Provide one configured transcript table workbook in the input directory or current run output directory. By default, DIAAD looks for:

```text
transcript_tables.xlsx
```

A common input layout is:

```text
diaad_data/input/
  transcript_tables/
    transcript_tables.xlsx
```

## Primary Outputs

By default, the command writes:

```text
diaad_data/output/diaad_YYMMDD_HHMM/chat_files/*.cha
diaad_data/output/diaad_YYMMDD_HHMM/transcript_tables/transcript_tables.xlsx
```

The output transcript table copy includes a `derived_file` column that records the generated CHAT filename for each exported sample.

## Immediate Next Step

Open the generated `.cha` files and confirm that speaker labels, utterance order, and comments look right. Treat this as a revision-export step, not a promise of lossless reconstruction of every original CHAT formatting detail.

## Read Next

- Transcript tabularization feature: `docs/manual/03_features/01_transcript_tabularization.md`
- Exact file name matching: `docs/manual/03_features/03_exact_file_name_matching.md`
- Transcripts module quickstart: `docs/manual/04_modules/01_transcripts/01_quickstart.md`
