# `words analyze` Quickstart

`diaad words analyze` summarizes a completed word-counting workbook by utterance and by sample.

## Run

```bash
diaad words analyze --config config
```

## Minimum Input

Provide a completed primary word-counting workbook. By default, DIAAD looks for:

```text
word_counting.xlsx
```

A common input layout is:

```text
diaad_data/input/
  word_counts/
    word_counting.xlsx
```

## Primary Outputs

By default, the command writes:

```text
word_count_analysis/
  word_counting_by_utterance.xlsx
  word_counting_by_sample.xlsx
```

If analysis-stage blinding is active, the command may also write a blind codebook and blinding diagnostics.

## Immediate Next Step

Inspect `word_counting_by_sample.xlsx` for missing counts, implausible totals, and samples affected by speaker exclusion before using the summary in later rate calculations.

## Read Next

- `words rates` quickstart: `docs/manual/04_modules/04_word_counting/05_commands/05_rates/01_quickstart.md`
- Word Counting module quickstart: `docs/manual/04_modules/04_word_counting/01_quickstart.md`
- Blinding module quickstart: `docs/manual/04_modules/08_blinding/01_quickstart.md`
- Generated Example I/O: `docs/manual/03_features/04_generated_example_io.md`
