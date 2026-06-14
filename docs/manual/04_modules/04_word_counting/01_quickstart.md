# Word Counting Module Quickstart

The Word Counting module supports manual or human-reviewed word-count workflows. DIAAD can create first-pass counts, generate reliability material, evaluate agreement, analyze completed coding, and calculate rates.

## Commands

| Command | Main use |
|---|---|
| `diaad words files` | Create word-count coding and reliability workbooks. |
| `diaad words evaluate` | Evaluate word-count reliability. |
| `diaad words reselect` | Select replacement word-count reliability material. |
| `diaad words analyze` | Summarize completed word counts. |
| `diaad words rates` | Calculate words per minute from sample summaries and speaking-time values. |

## Typical Sequence

```text
transcripts tabularize
words files
human review of word counts
words evaluate
words analyze
templates times
words rates
```

If CU coding output is available, `words files` can use it as preferred input so non-countable utterances can be handled more explicitly. Otherwise transcript tables can serve as the source.

## Common Outputs

| Step | Typical outputs |
|---|---|
| File generation | `word_counts/word_counting.xlsx`, `word_count_reliability.xlsx`, optional blind codebook |
| Reliability evaluation | `word_count_reliability/word_count_reliability_results.xlsx`, report |
| Analysis | `word_count_analysis/word_counting_by_utterance.xlsx`, `word_counting_by_sample.xlsx` |
| Rates | `word_count_analysis/word_counting_rates.xlsx` |

## Read Next

- Word Counting Versus Target Vocabulary Coverage: `docs/manual/03_features/02_word_counting_vs_target_vocabulary_coverage.md`
- Transcript tabularization: `docs/manual/03_features/01_transcript_tabularization.md`
- Configuration: `docs/manual/02_operation/04_configuration.md`

Later command pages describe workbook columns, first-pass counting behavior, reliability evaluation, and rates.
