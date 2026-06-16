# Transcript Text Normalization and Speaker Exclusion Usage Guide

Text normalization and speaker exclusion are shared settings, but they do not affect every command in the same way. Use them deliberately and report them when they influence analytic results.

## Normalization Settings

The normalization settings primarily affect transcript comparison, especially `transcripts evaluate`.

`strip_clan: true` removes common CHAT and CLAN markup before comparison. This helps reliability metrics focus on the spoken text instead of transcription notation.

`prefer_correction: true` uses corrected forms when CLAN correction notation is present. Set it to `false` only when the original uncorrected token should be retained for the comparison.

`lowercase: true` removes casing as a difference source. Set it to `false` if capitalization is meaningful for your comparison.

These settings process text for analysis. They do not rewrite the source `.cha` files or the transcript table workbook.

## Speaker Exclusion

Use `exclude_speakers` when a workflow should ignore selected participant tiers or speaker labels.

Example:

```yaml
project:
  exclude_speakers:
    - INV
    - CLN
```

For CHAT-based transcription reliability evaluation, excluded labels are compared to CHAT participant tier codes.

For transcript-table workflows that support speaker exclusion, DIAAD filters rows by the `speaker` column. Those row filters are case-insensitive after trimming whitespace.

## When To Exclude Speakers

Speaker exclusion is most useful when the analytic unit is a target participant's language rather than the whole interaction. For example, a clinician-client dialog may include prompts, clarifications, and task instructions from a clinician. If a word-count or target-vocabulary analysis is meant to describe the client's speech, excluding the clinician speaker label prevents those prompts from entering the count.

Do not exclude a speaker just because they are not the primary participant if their speech is part of the construct being analyzed. For some dialog and turn-taking analyses, partner speech may be analytically central.

## Practical Checks

Before running downstream analysis:

- inspect the transcript table `speaker` values;
- confirm that configured labels match the labels actually present;
- decide whether exclusions should apply to transcript reliability, coding-file generation, analysis, or only selected modules;
- keep the setting consistent across comparable runs.

Changing speaker exclusion after coding or analysis can change denominators, counts, coverage rates, and reliability comparisons.

## Read Next

- Configuration operation page: `docs/manual/02_operation/04_configuration.md`
- Transcript preprocessing: `docs/manual/05_functionalities/06_transcript_preprocessing_tabularization_chat_export/02_usage_guide.md`
- Speaking-time rate calculation: `docs/manual/05_functionalities/15_speaking_time_rate_calculation/02_usage_guide.md`
