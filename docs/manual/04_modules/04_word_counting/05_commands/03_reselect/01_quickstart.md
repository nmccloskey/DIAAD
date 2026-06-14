# `words reselect` Quickstart

`diaad words reselect` creates replacement word-count reliability material after an earlier reliability workbook has already been used.

## Run

```bash
diaad words reselect --config config
```

## Minimum Inputs

Provide prior word-counting and reliability workbooks:

```text
word_counting.xlsx
word_count_reliability.xlsx
```

A common input layout is:

```text
diaad_data/input/
  word_counts/
    word_counting.xlsx
    word_count_reliability.xlsx
```

## Primary Output

By default, the command writes:

```text
reselected_word_count_reliability/
  reselected_word_count_reliability.xlsx
```

## Immediate Next Step

Review the selected samples and distribute `reselected_word_count_reliability.xlsx` only after confirming that the rows are genuinely eligible replacement reliability material.

## Read Next

- `words files` usage guide: `docs/manual/04_modules/04_word_counting/05_commands/01_files/02_usage_guide.md`
- `words evaluate` quickstart: `docs/manual/04_modules/04_word_counting/05_commands/02_evaluate/01_quickstart.md`
- Configuration: `docs/manual/02_operation/04_configuration.md`
- Generated Example I/O: `docs/manual/03_features/04_generated_example_io.md`
