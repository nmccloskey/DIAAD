# Speaking-Time Rate Calculation Usage Guide

Speaking-time rates normalize count-like measures by the amount of speech represented in each sample. They are useful when samples differ in length or when a project wants measures such as complete utterances per minute or words per minute.

## Speaking-Time File

Use `templates times` to create:

```text
coding_templates/speaking_times.xlsx
```

The workbook contains one row per sample and a blank `speaking_time` column. Enter speaking time in seconds.

The shared rate utility later standardizes this table by:

1. reading the configured sample identifier column;
2. reading the configured speaking-time column;
3. coercing speaking time to numeric;
4. summing duplicate sample IDs with a warning;
5. calculating `speaking_minutes = speaking_time / 60`.

Entering minutes instead of seconds will make rates too small by a factor of 60.

## Module-Specific Numerators

Rate commands use shared per-minute math but different numerators.

| Command | Rate behavior |
|---|---|
| `cus rates` | Rates `cu`, `p_sv`, and `p_rel` per minute from the Complete Utterances sample summary. |
| `words rates` | Rates `total_words` per minute from the word-count sample summary. |
| `powers rates` | Rates numeric POWERS dialog-summary count-like columns, skipping proportions and ratios. |
| `vocab rates` | Rates count-like target-vocabulary summary columns and preserves coverage, percentile, and existing rate fields as context. |

For Target Vocabulary Coverage, `vocab analyze` already writes `speaking_time` and `core_tokens_per_min` in the analysis summary when speaking-time data are available in the input. `vocab rates` reads one or more `target_vocab_data_*.xlsx` analysis workbooks and adds additional inferred per-minute columns for eligible numeric count-like fields.

## File Placement

For CU, Word Counting, and POWERS rate commands, place the completed speaking-time workbook where the rate command can find it, usually in the active input tree alongside the relevant analysis or summary workbook.

For TVC rate commands, ensure the `target_vocab_data_*.xlsx` analysis workbook contains the expected `summary` sheet and `speaking_time` column.

## Interpreting Rates

Rates are per minute, but the unit depends on the numerator:

- complete utterances per minute;
- propositions or relevant content units per minute;
- words per minute;
- POWERS dialog-summary counts per minute;
- target-vocabulary token or base-form measures per minute.

Do not compare rates across modules as if they measure the same construct.

## Read Next

- Word Counting versus Target Vocabulary Coverage: `docs/manual/03_features/02_word_counting_vs_target_vocabulary_coverage.md`
- Transcript text normalization and speaker exclusion: `docs/manual/05_functionalities/07_transcript_text_normalization_speaker_exclusion/02_usage_guide.md`
- Configuration: `docs/manual/02_operation/04_configuration.md`
