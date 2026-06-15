# `vocab rates` Quickstart

`diaad vocab rates` adds per-minute rates to target-vocabulary analysis summaries.

## Run

```bash
diaad vocab rates --config config
```

## Minimum Input

Provide at least one target-vocabulary analysis workbook:

```text
target_vocab_data_YYMMDD_HHMM.xlsx
```

A common input layout is:

```text
diaad_data/input/
  target_vocab/
    target_vocab_data_YYMMDD_HHMM.xlsx
```

The command uses `speaking_time` from the analysis workbook's `summary` sheet. It does not read a separate speaking-time template.

## Primary Output

By default, the command writes:

```text
target_vocab/
  target_vocab_rates.xlsx
```

## Immediate Next Step

Confirm that `speaking_time` values are in seconds and that per-minute columns were added for the expected count-like fields.

## Read Next

- `vocab analyze` quickstart: `docs/manual/04_modules/06_target_vocabulary_coverage/05_commands/03_analyze/01_quickstart.md`
- Target Vocabulary Coverage research context: `docs/manual/04_modules/06_target_vocabulary_coverage/03_research_context.md`
- Generated Example I/O: `docs/manual/03_features/04_generated_example_io.md`
