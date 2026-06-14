# `powers files` Usage Guide

Use `diaad powers files` after transcript tabularization, when utterance-level dialogue data are ready for POWERS coding.

## Important Settings

| Setting | Default | Effect |
|---|---|---|
| `project.reliability_fraction` | `0.2` | Fraction of samples selected for the reliability workbook. |
| `project.num_coders` | `0` | Controls generated coder IDs. |
| `project.exclude_speakers` | `[]` | Speaker labels initialized as not applicable for Section C fields before automation. |
| `project.automate_powers` | `true` | Enables automated first-pass support for selected fields. |
| `advanced.powers_coding_filename` | `powers_coding.xlsx` | Primary coding workbook filename. |
| `advanced.powers_reliability_filename` | `powers_reliability_coding.xlsx` | Reliability workbook filename. |
| `advanced.spacy_model_name` | `en_core_web_sm` | spaCy model used for NLP-backed automation. |
| `advanced.sample_id_column` | `sample_id` | Sample identifier column. |
| `advanced.utterance_id_column` | `utterance_id` | Utterance identifier column. |
| `advanced.auto_blind` | `false` | Whether supported coding exports should blind configured columns. |

## Workbook Structure

The primary workbook contains:

```text
utterance_coding
section_e
```

The `utterance_coding` sheet contains utterance-level transcript columns, `coder_id`, `POWERS_comment`, POWERS coding fields, and any automation helper columns.

The `section_e` sheet contains one row per sample with:

```text
type_of_day
amount_of_enjoyment
degree_of_difficulty
other_notes
```

These Section E fields are carried as sample-level note or descriptor fields. In the current implementation, they are not summarized by `powers analyze`, evaluated by `powers evaluate`, or converted to rates by `powers rates`.

## Automated Fields

POWERS automation is first-pass support only. When enabled and available, DIAAD fills:

| Field | Automated support |
|---|---|
| `speech_units` | Count derived from cleaned utterance text, excluding built-in unintelligible placeholders. |
| `filled_pauses` | Count of filler forms such as `um` and `uh`. |
| `content_words` | spaCy-assisted content-word count. |
| `num_nouns` | spaCy-assisted noun/proper-noun count. |
| `tagged_utterance` | Helper column inserted after `utterance` to show token tags used for review. |

The command does not automatically code:

```text
turn_type
circumlocutions
sem_paras
phon_errs
neologisms
comments
lg_pauses
collab_repair
section_e fields
```

Review automated fields before treating them as coded values. If the spaCy model cannot be loaded or processing fails, DIAAD logs the problem and leaves the workbook available for manual coding.

## Coder Assignment

`num_coders` controls the generated `coder_id` values:

| `num_coders` | Behavior |
|---|---|
| `0` | Blank coder IDs. |
| `1` | Primary and reliability rows use coder ID `1`. |
| `2` or more | Primary samples are distributed across coder IDs; reliability rows rotate to another coder ID when possible. |

Reliability selection is sample-based. If a sample is selected for reliability, all of its utterance rows are included.

## Common Problems

If the command cannot find transcript tables, run `diaad transcripts tabularize` first or check `advanced.transcript_table_filename`.

If automation columns are blank, check `project.automate_powers`, the installed NLP dependencies, and `advanced.spacy_model_name`.

If Section E values do not appear in analysis output, that is expected in the current implementation. Section E is generated for human-entered sample-level notes or descriptors, not for the automated POWERS analysis summaries.
