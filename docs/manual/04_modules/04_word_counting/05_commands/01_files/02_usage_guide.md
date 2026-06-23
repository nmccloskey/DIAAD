# `words files` Usage Guide

Use `diaad words files` when utterance-level data are ready to become human-review word-count workbooks.

## Input Priority

The command first looks for:

```text
cu_coding_by_utterance.xlsx
```

When this file is available, DIAAD can use completed CU information to mark clearly non-countable rows as `NA`. This is usually preferable after a CU workflow because the word-count file can inherit earlier utterance inclusion decisions.

If CU output is not available, DIAAD uses the configured transcript table workbook, usually:

```text
transcript_tables.xlsx
```

In that fallback mode, the command prepares word-count rows directly from transcript utterances.

## Important Settings

| Setting | Default | Effect |
|---|---|---|
| `project.reliability_fraction` | `0.2` | Fraction of samples selected for the reliability workbook. |
| `project.num_coders` | `0` | Controls generated coder IDs. |
| `project.random_seed` | `99` | Seed used for sample shuffling and reliability selection. |
| `project.exclude_speakers` | `[]` | Speaker labels marked `NA` for word counting. |
| `advanced.sample_id_column` | `sample_id` | Sample identifier column. |
| `advanced.utterance_id_column` | `utterance_id` | Utterance identifier column. |
| `advanced.transcript_table_filename` | `transcript_tables.xlsx` | Fallback transcript table filename. |
| `advanced.auto_blind` | `false` | Whether supported coding exports should blind configured columns. |

## First-Pass Counts

DIAAD seeds the `word_count` column with an automated first-pass count. The counter handles common transcript features such as contractions, integers, CHAT-like annotations, and filler tokens.

This first-pass count is not a substitute for project-specific coding. Human reviewers should still inspect rows where utterance boundaries, fillers, repetitions, symbols, unintelligible speech, off-task talk, or protocol-specific exclusion rules matter.

Rows from excluded speakers are assigned `NA`. When the input is CU-derived, rows with only neutral CU values are also assigned `NA`.

## Coder Modes

`num_coders` controls the generated `coder_id` values:

| `num_coders` | Behavior |
|---|---|
| `0` | Primary and reliability rows have blank coder IDs. |
| `1` | Primary and reliability rows use coder ID `1`. |
| `2` or more | Samples are assigned across available coder IDs; reliability rows receive an alternate coder when possible. |

Reliability selection is sample-based. If a sample is selected for reliability, all of its utterance rows are included in the reliability workbook.

## Blinding

When `advanced.auto_blind` is true, generated word-count workbooks can blind configured identifier columns and write `word_count_blind_codebook.xlsx`. Protect this codebook because later analysis may need it to reconnect blinded sample identifiers.

## Common Problems

If the command uses transcript tables when you expected CU-derived input, check whether `cu_coding_by_utterance.xlsx` is present under the input directory or current run output directory.

If many rows are `NA`, check `project.exclude_speakers` and, for CU-derived input, the CU columns used to determine whether a row is countable.

If first-pass counts do not match the protocol, use the workbook as intended: revise the counts manually before running `words analyze`.
