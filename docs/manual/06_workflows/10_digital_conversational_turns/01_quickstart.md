# Digital Conversational Turns Quickstart

Use Digital Conversational Turns, or DCT, when a project wants compact turn-sequence data about conversational participation and speaker transitions. DCT is conceptually separate from POWERS: it describes turn dynamics rather than transcript-derived lexical or error variables.

## Choose An Input Mode

Manual DCT mode uses completed turn-string workbooks:

```text
conversation_turns.xlsx
conversation_turns_reliability.xlsx
```

Transcript-table mode uses speaker sequences from a reviewed transcript table when no primary DCT workbook is found.

## Manual DCT Sequence

After primary and reliability DCT workbooks are completed, evaluate reliability:

```bash
diaad turns evaluate
```

Then analyze the primary workbook:

```bash
diaad turns analyze
```

## Transcript-Table Sequence

If a transcript table is available and no `conversation_turns.xlsx` workbook is found, run:

```bash
diaad turns analyze
```

DIAAD falls back to ordered transcript-table speaker tags and writes a transcript-derived turns analysis workbook.

## Key Reminder

The current CLI registry exposes:

```text
turns evaluate
turns analyze
```

Prepare DCT coding workbooks outside the current `turns` command surface.

## Read Next

- Digital Conversational Turns module: `docs/manual/04_modules/07_digital_conversational_turns/01_quickstart.md`
- `turns evaluate`: `docs/manual/04_modules/07_digital_conversational_turns/05_commands/02_evaluate/01_quickstart.md`
- `turns analyze`: `docs/manual/04_modules/07_digital_conversational_turns/05_commands/04_analyze/01_quickstart.md`
- Exact file name matching: `docs/manual/03_features/03_exact_file_name_matching.md`
