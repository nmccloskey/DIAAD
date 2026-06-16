# Reliability Selection, Evaluation, and Reselection Implementation Notes

Reliability behavior is spread across transcript reliability utilities, module-specific evaluators, shared sampling helpers, and shared reselection helpers.

## Source Anchors

Primary sources:

- `src/diaad/transcripts/transcription_reliability_selection.py`
- `src/diaad/transcripts/transcription_reliability_evaluation.py`
- `src/diaad/coding/utils/sampling.py`
- `src/diaad/coding/utils/rel_eval_utils.py`
- `src/diaad/coding/utils/reselection_utils.py`
- `src/diaad/coding/compl_utts/rel_evaluation.py`
- `src/diaad/coding/compl_utts/rel_reselection.py`
- `src/diaad/coding/word_counts/rel_evaluation.py`
- `src/diaad/coding/word_counts/rel_reselection.py`
- `src/diaad/coding/powers/rel_evaluation.py`
- `src/diaad/coding/powers/rel_reselection.py`
- `src/diaad/coding/convo_turns/rel_evaluation.py`
- `src/diaad/coding/convo_turns/rel_reselection.py`

Relevant tests:

- `tests/test_coding/test_utils/test_sampling.py`
- `tests/test_coding/test_utils/test_reselection_utils.py`
- module-specific reliability tests where present

## Subset Size

`calc_subset_size()` returns:

```text
max(1, ceil(reliability_fraction * n_samples))
```

It validates that the fraction is in `(0, 1]` and that the sample set is non-empty.

## Transcription Reliability

`select_transcription_reliability_samples()` writes:

```text
transcription_reliability_selection/transcription_reliability_samples.xlsx
```

The workbook includes:

- `reliability_selection`
- `all_transcripts`

When CHAT objects are available, the selector also writes blank `*_reliability.cha` files containing CHAT headers.

`evaluate_transcription_reliability()` writes:

```text
transcription_reliability_evaluation/transcription_reliability_evaluation.xlsx
transcription_reliability_evaluation/transcription_reliability_report.txt
transcription_reliability_evaluation/global_alignments/
```

It can identify reliability files by configured tag or by files inside the configured reliability directory. It computes token and character count differences, Levenshtein distance/similarity, Needleman-Wunsch score/normalized score, and alignment text files.

## Module Evaluators

Complete Utterances reliability computes categorical agreement and kappa at the utterance level, plus ICC summaries for sample-level totals.

Word Counting reliability computes paired utterance differences, percent differences, percent similarity, agreement flags, and ICC(2,1).

POWERS reliability separates continuous and categorical fields. Continuous fields receive difference, percent-difference, agreement, variance, and ICC diagnostics where possible. Categorical fields receive percent agreement and Cohen's kappa.

Digital Conversational Turns reliability compares turn counts and turn sequences, including percent agreement, ICC where possible, Levenshtein sequence metrics, and alignment files.

## Reselection

Shared reselection utilities pair original and reliability files using configured metadata fields when available. They collect sample IDs already represented in prior reliability files, select unused IDs from the original workbook, and write replacement reliability workbooks under module-specific `reselected_*` directories.

`turns reselect` is not currently registered in `src/diaad/cli/commands.py`, even though source helpers for DCT reselection exist.

## Read Next

- General subsetting implementation notes: `docs/manual/05_functionalities/13_sample_subsetting_resubsetting/04_implementation_notes.md`
- Configurable identifiers implementation notes: `docs/manual/05_functionalities/10_configurable_sample_utterance_identifiers/04_implementation_notes.md`
- Command-line operation: `docs/manual/02_operation/02_command_line.md`
