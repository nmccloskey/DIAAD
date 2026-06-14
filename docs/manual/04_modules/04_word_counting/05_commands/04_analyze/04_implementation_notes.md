# `words analyze` Implementation Notes

`words analyze` dispatches to `analyze_word_counts()`.

## Dispatch Path

The main path is:

1. `src/diaad/cli/commands.py` registers `words analyze`.
2. `src/diaad/cli/dispatch.py` dispatches it as a word-counting command.
3. `src/diaad/core/run_context.py` threads input/output paths, word-count filename and column, blinding config, sample identifier, and excluded speakers.
4. `src/diaad/core/run_wrappers.py` calls `analyze_word_counts()`.
5. `src/diaad/coding/word_counts/analysis.py` writes analysis outputs.

## Input Discovery

The command uses exact filename discovery for the configured word-counting workbook. The default is:

```text
word_counting.xlsx
```

The configured word-count column default is:

```text
word_count
```

## Summary Logic

Administrative columns such as `id`, `comment`, and `wc_comment` are removed from the utterance-level output when present.

The configured word-count column is coerced to numeric values. Sample-level summaries count nonmissing and missing values separately, then calculate total, mean, standard deviation, minimum, and maximum word counts per sample.

When a `speaker` column is present, rows from `project.exclude_speakers` are dropped before summary.

## Blinding

Analysis can unblind earlier word-count outputs when a compatible codebook is available. It can also apply analysis-stage blinding to configured columns and write diagnostics.

## Relevant Sources

- `src/diaad/coding/word_counts/analysis.py`
- `src/diaad/coding/utils/transcript.py`
- `tests/test_coding/test_word_counts/test_identifiers.py`
