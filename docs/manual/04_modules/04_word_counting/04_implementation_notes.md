# Word Counting Implementation Notes

The Word Counting module is implemented under `src/diaad/coding/word_counts/`.

## File Generation

`words files` prefers completed CU-derived input when it is available, because CU coding can indicate which utterances should not receive ordinary counts. If that input is not found, the command falls back to transcript tables.

The first-pass counter normalizes text, handles some CHAT-like annotations, expands selected forms, and excludes configured speakers. Rows that should not be counted can be written as `NA` rather than numeric counts.

Primary and reliability outputs are written under `word_counts/`. Optional blinding can produce a blind codebook alongside coding files.

## Analysis

`words analyze` reads the configured word-count coding file, coerces counts to numeric values where possible, keeps missing/non-counted rows distinct, and writes:

- `word_counting_by_utterance.xlsx`
- `word_counting_by_sample.xlsx`

The sample summary includes total words and utterance-level descriptive summaries.

## Reliability And Rates

`words evaluate` merges primary and reliability workbooks on sample and utterance identifiers, computes count differences and agreement fields, and writes reliability results plus a report.

`words rates` reads `word_counting_by_sample.xlsx` and `speaking_times.xlsx`, converts seconds to minutes, and writes `word_counting_rates.xlsx`.

## Boundaries

The first-pass counter is a convenience layer. Command and research-context pages should continue to emphasize human review when protocol rules require judgment.
