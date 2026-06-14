# `words files` Implementation Notes

`words files` dispatches to `make_word_count_files()`.

## Dispatch Path

The main path is:

1. `src/diaad/cli/commands.py` registers `words files`.
2. `src/diaad/cli/dispatch.py` dispatches it as a word-counting command.
3. `src/diaad/core/run_context.py` threads reliability settings, identifier columns, excluded speakers, blinding config, and transcript-table filename.
4. `src/diaad/core/run_wrappers.py` calls `make_word_count_files()`.
5. `src/diaad/coding/word_counts/files.py` writes the workbooks.

## Input Discovery

The command prefers the exact CU analysis filename:

```text
cu_coding_by_utterance.xlsx
```

If no CU by-utterance workbook is found, it falls back to the configured transcript table filename.

## Output Shape

The prepared workbook is limited to these columns when present or generated:

```text
sample_id
utterance_id
speaker
utterance
comment
id
word_count
wc_comment
```

Custom sample and utterance identifier names configured in `advanced.sample_id_column` and `advanced.utterance_id_column` replace `sample_id` and `utterance_id`.

## Countability And First-Pass Counting

For transcript-table input, countability is based on speaker exclusion and utterance text.

For CU-derived input, the command also checks CU columns. Rows with only neutral CU values are assigned `NA`.

The `count_words()` helper normalizes utterance text before counting. It expands selected contractions, converts integer tokens to words, strips or simplifies common annotation forms, and filters a small list of filler or placeholder tokens. This behavior is intentionally heuristic; the coding workbook remains a human-review artifact.

## Reliability Selection

Reliability rows are selected by sample block. The command uses the shared sampling and coder-assignment utilities so that all utterances for a selected sample stay together.

## Relevant Sources

- `src/diaad/coding/word_counts/files.py`
- `src/diaad/coding/utils/coders.py`
- `src/diaad/coding/utils/sampling.py`
- `src/diaad/coding/utils/transcript.py`
- `tests/test_coding/test_word_counts/test_files.py`
- `tests/test_coding/test_word_counts/test_identifiers.py`
