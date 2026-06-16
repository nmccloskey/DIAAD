# POWERS Automation Support Implementation Notes

POWERS automation is implemented in `diaad.coding.powers.automation` and called from POWERS file-generation and reselection paths.

## Source Anchors

Primary sources:

- `src/diaad/coding/powers/automation.py`
- `src/diaad/coding/powers/files.py`
- `src/diaad/coding/powers/rel_reselection.py`
- `src/diaad/coding/powers/analysis.py`
- `src/diaad/coding/powers/evaluation.py`
- `src/diaad/coding/powers/rates.py`
- `src/diaad/core/config.py`
- `src/diaad/core/run_context.py`

Relevant tests:

- `tests/test_coding/test_powers/test_automation.py`
- `tests/test_coding/test_powers/test_identifiers.py`

## Configuration

The default configuration enables automation:

```yaml
project:
  automate_powers: true

advanced:
  spacy_model_name: en_core_web_sm
```

The model is loaded through `psair.nlp.NLPModel`. If model loading fails, `run_automation()` logs the error and returns the dataframe unchanged.

## Automated Field Path

`run_automation()` fills:

```text
speech_units
filled_pauses
content_words
num_nouns
tagged_utterance
```

`speech_units` and `filled_pauses` are rule-based over utterance text. `content_words`, `num_nouns`, and `tagged_utterance` use spaCy-backed token information plus DIAAD filtering rules.

Content-word rules exclude built-in generic terms, common unintelligible placeholders, selected CHAT-like forms, auxiliaries, and modal auxiliaries. They include nouns, proper nouns, main verbs, adjectives, selected adverbs, and numerals according to the current helper rules.

## Command Integration

`powers files` prepares the primary and reliability workbooks, initializes POWERS columns, and applies automation when `project.automate_powers` is true.

`powers reselect` loads the original POWERS coding workbook, selects unused reliability samples, clears manual POWERS fields for the selected rows, and reapplies automation when enabled.

`powers analyze`, `powers evaluate`, and `powers rates` consume completed coding outputs. They do not run `run_automation()`.

## Section E Boundary

`powers files` writes a `section_e` sheet with one row per sample and these fields:

```text
type_of_day
amount_of_enjoyment
degree_of_difficulty
other_notes
```

The current analysis implementation reads the utterance-level coding sheet and computes utterance, turn, speaker, and dialog summaries from utterance-level fields. Section E is not read by the analysis path and is not part of the current reliability or rate paths.

## Source Concern For Review

`powers files` initializes Section C fields as not applicable for configured excluded speakers before automation runs. The current automation pass then fills automated fields from utterance text. Review whether automated fields should remain blank or not applicable for excluded-speaker rows in the final intended behavior.

## Read Next

- `powers files` implementation notes: `docs/manual/04_modules/05_powers/05_commands/01_files/04_implementation_notes.md`
- `powers reselect` implementation notes: `docs/manual/04_modules/05_powers/05_commands/05_reselect/04_implementation_notes.md`
- `powers analyze` implementation notes: `docs/manual/04_modules/05_powers/05_commands/02_analyze/04_implementation_notes.md`
- Configuration: `docs/manual/02_operation/04_configuration.md`
