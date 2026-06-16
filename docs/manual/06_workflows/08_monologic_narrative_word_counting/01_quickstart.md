# Monologic Narrative Word Counting Quickstart

Use this workflow when monologic narrative samples need human-reviewed word counts. DIAAD can seed first-pass counts, generate reliability material, summarize completed counts, and calculate words per minute.

## Starting Point

Word Counting can start from transcript tables, but it is often cleaner after Complete Utterances analysis when CU outputs identify rows that should not be counted.

Preferred after CU:

```bash
diaad cus analyze
diaad words files
```

Fallback from transcript tables:

```bash
diaad transcripts tabularize
diaad words files
```

## Core Sequence

Review the first-pass counts manually, then run:

```bash
diaad words evaluate
diaad words analyze
```

Use reselection only if another reliability round is needed:

```bash
diaad words reselect
```

If rates are needed:

```bash
diaad templates times
diaad words rates
```

## Key Outputs

```text
word_counts/word_counting.xlsx
word_counts/word_count_reliability.xlsx
word_count_reliability/
word_count_analysis/
```

## Important Distinction

Word Counting estimates countable language quantity. Target Vocabulary Coverage estimates production of a configured target lexicon. Both involve words, but they answer different questions.

## Read Next

- Word Counting module: `docs/manual/04_modules/04_word_counting/01_quickstart.md`
- Word Counting versus Target Vocabulary Coverage: `docs/manual/03_features/02_word_counting_vs_target_vocabulary_coverage.md`
- `words files`: `docs/manual/04_modules/04_word_counting/05_commands/01_files/01_quickstart.md`
- Speaking-time rates: `docs/manual/05_functionalities/15_speaking_time_rate_calculation/02_usage_guide.md`
