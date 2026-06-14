# `words evaluate` Quickstart

`diaad words evaluate` compares completed primary and reliability word-count workbooks.

## Run

```bash
diaad words evaluate --config config
```

## Minimum Inputs

Provide completed primary and reliability workbooks:

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

## Primary Outputs

By default, the command writes:

```text
word_count_reliability/
  word_count_reliability_results.xlsx
  word_count_reliability_report.txt
```

## Immediate Next Step

Read the text report for coverage and summary metrics, then inspect `word_count_reliability_results.xlsx` for utterance-level disagreements.

## Read Next

- `words evaluate` research context: `docs/manual/04_modules/04_word_counting/05_commands/02_evaluate/03_research_context.md`
- Word Counting module research context: `docs/manual/04_modules/04_word_counting/03_research_context.md`
- Exact file-name matching: `docs/manual/03_features/03_exact_file_name_matching.md`
- Configuration: `docs/manual/02_operation/04_configuration.md`
- Generated Example I/O: `docs/manual/03_features/04_generated_example_io.md`
