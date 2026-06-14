# `words files` Quickstart

`diaad words files` creates word-count coding and reliability workbooks from utterance-level DIAAD data.

## Run

```bash
diaad words files --config config
```

## Minimum Inputs

The preferred input is a completed Complete Utterance by-utterance workbook:

```text
cu_coding_by_utterance.xlsx
```

If that file is not available, DIAAD falls back to the configured transcript table workbook. By default, that file is:

```text
transcript_tables.xlsx
```

A common input layout is:

```text
diaad_data/input/
  cu_coding_analysis/
    cu_coding_by_utterance.xlsx
```

or:

```text
diaad_data/input/
  transcript_tables/
    transcript_tables.xlsx
```

## Primary Outputs

By default, the command writes:

```text
word_counts/
  word_counting.xlsx
  word_count_reliability.xlsx
  word_count_blind_codebook.xlsx
```

The blind codebook is written only when configured blinding is active.

## Immediate Next Step

Open `word_counting.xlsx` and treat the `word_count` values as first-pass coding support. Confirm them against the project's word-counting rules before using the workbook for analysis.

## Read Next

- Word Counting module quickstart: `docs/manual/04_modules/04_word_counting/01_quickstart.md`
- Word Counting research context: `docs/manual/04_modules/04_word_counting/03_research_context.md`
- Complete Utterances module quickstart: `docs/manual/04_modules/03_complete_utterances/01_quickstart.md`
- Configuration: `docs/manual/02_operation/04_configuration.md`
- Generated Example I/O: `docs/manual/03_features/04_generated_example_io.md`
