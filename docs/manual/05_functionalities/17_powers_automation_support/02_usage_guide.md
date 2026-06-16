# POWERS Automation Support Usage Guide

POWERS automation is a first-pass aid for fields that can be estimated from utterance text. It works best when users treat the generated values as prefilled coding work, review them against the POWERS protocol, and correct them before analysis.

## Settings

The main settings are:

```yaml
project:
  automate_powers: true

advanced:
  spacy_model_name: en_core_web_sm
```

Set `project.automate_powers` to `false` if the project will complete all supported fields manually or if the NLP dependency path is not available in the working environment.

Change `advanced.spacy_model_name` only when the chosen environment has another compatible spaCy model installed and the project has reviewed the effect of that model on POWERS first-pass values.

## Automated Fields

| Field | First-pass behavior |
|---|---|
| `speech_units` | Counts cleaned utterance tokens while excluding built-in unintelligible placeholders. |
| `filled_pauses` | Counts filler forms such as `um` and `uh` from utterance text. |
| `content_words` | Uses spaCy-assisted token tags and DIAAD rules to count content words. |
| `num_nouns` | Uses spaCy-assisted token tags to count nouns and proper nouns. |
| `tagged_utterance` | Adds a helper representation of tokens and tags for human review. |

The helper `tagged_utterance` column is not a POWERS score. It is included to make automated counts easier to inspect.

## Manual Fields

The following fields remain manual in the current implementation:

```text
turn_type
circumlocutions
sem_paras
phon_errs
neologisms
comments
lg_pauses
collab_repair
POWERS_comment
```

The Section E sheet contains:

```text
type_of_day
amount_of_enjoyment
degree_of_difficulty
other_notes
```

These Section E fields are carried as sample-level note or descriptor fields. They are not part of the current POWERS analysis, reliability evaluation, or rate-calculation paths.

## Review Workflow

A typical human-reviewed workflow is:

1. Run `diaad powers files`.
2. Inspect automated fields and `tagged_utterance`.
3. Correct automated values where needed.
4. Complete the manual POWERS fields.
5. Complete independent reliability coding.
6. Run `diaad powers evaluate` and inspect agreement.
7. Run `diaad powers analyze` and `diaad powers rates` only after the coding is ready for analysis.

If `diaad powers reselect` is used later, it clears prior manual POWERS fields for the newly selected rows and reapplies first-pass automation when `project.automate_powers` is true.

## Reliability Caution

If both primary and reliability coders leave the same automated values unchanged, reliability statistics for those fields partly reflect shared automation rather than independent human coding. Decide in advance how automated values should be reviewed, corrected, and documented.

## Common Problems

If automated columns are blank, check that `project.automate_powers` is true, the NLP dependencies are installed, and the configured spaCy model is available.

If Section E fields do not appear in POWERS analysis, evaluation, or rate outputs, that is expected in the current implementation.

## Read Next

- `powers files` usage guide: `docs/manual/04_modules/05_powers/05_commands/01_files/02_usage_guide.md`
- `powers evaluate` research context: `docs/manual/04_modules/05_powers/05_commands/04_evaluate/03_research_context.md`
- Reliability selection, evaluation, and reselection: `docs/manual/05_functionalities/12_reliability_selection_evaluation_reselection/02_usage_guide.md`
- Testing: `docs/manual/02_operation/05_testing.md`
