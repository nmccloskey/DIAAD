# `turns evaluate` Quickstart

`diaad turns evaluate` compares primary and reliability-coded DCT turn strings.

## Run

```bash
diaad turns evaluate --config config
```

## Minimum Inputs

Place the completed primary and reliability workbooks where DIAAD can find them by exact filename:

```text
conversation_turns.xlsx
conversation_turns_reliability.xlsx
```

Each workbook must include the configured sample identifier column and `turns`. `session` and `bin` are recommended because they define the comparison unit.

## Primary Outputs

By default, the command writes:

```text
turns_reliability/
  conversation_turns_reliability_results.xlsx
  conversation_turns_reliability_report.txt
  global_alignments/
```

The results workbook contains:

```text
counts
sequences
samples
```

## Immediate Next Step

Inspect `counts` for speaker-level count agreement and `sequences` for Levenshtein similarity by sample/session/bin. Then read the plain-text report for coverage, ICC(2,1), and sequence-similarity bands.

## Read Next

- `turns evaluate` usage guide: `docs/manual/04_modules/07_digital_conversational_turns/05_commands/02_evaluate/02_usage_guide.md`
- `turns evaluate` research context: `docs/manual/04_modules/07_digital_conversational_turns/05_commands/02_evaluate/03_research_context.md`
- Digital Conversational Turns research context: `docs/manual/04_modules/07_digital_conversational_turns/03_research_context.md`
- Exact file-name matching: `docs/manual/03_features/03_exact_file_name_matching.md`
