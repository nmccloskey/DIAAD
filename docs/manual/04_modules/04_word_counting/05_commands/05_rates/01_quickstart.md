# `words rates` Quickstart

`diaad words rates` combines word-count sample summaries with speaking-time values to calculate words per minute.

## Run

```bash
diaad words rates --config config
```

## Minimum Inputs

Provide a word-count sample summary and a speaking-time workbook. Defaults are:

```text
word_counting_by_sample.xlsx
speaking_times.xlsx
```

A common input layout is:

```text
diaad_data/input/
  word_count_analysis/
    word_counting_by_sample.xlsx
  speaking_times/
    speaking_times.xlsx
```

## Primary Output

By default, the command writes:

```text
word_count_analysis/
  word_counting_rates.xlsx
```

## Immediate Next Step

Check that `speaking_time` values were entered in seconds and that `total_words_per_min` is present for the expected samples.

## Read Next

- `words analyze` quickstart: `docs/manual/04_modules/04_word_counting/05_commands/04_analyze/01_quickstart.md`
- Speaking-time templates: `docs/manual/04_modules/02_templates/05_commands/03_times/01_quickstart.md`
- Word Counting module quickstart: `docs/manual/04_modules/04_word_counting/01_quickstart.md`
- Generated Example I/O: `docs/manual/03_features/04_generated_example_io.md`
