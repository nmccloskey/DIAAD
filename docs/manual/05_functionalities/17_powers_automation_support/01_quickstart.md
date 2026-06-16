# POWERS Automation Support Quickstart

POWERS automation provides first-pass values for selected POWERS fields. It is intended to reduce repetitive coding work, not to replace human coding or protocol review.

Automation is enabled by default:

```yaml
project:
  automate_powers: true
advanced:
  spacy_model_name: en_core_web_sm
```

If the configured spaCy model cannot be loaded, DIAAD logs the problem and still leaves the workbook available for manual coding.

## Where Automation Runs

Automation runs when supported POWERS workbooks are generated or regenerated:

```bash
diaad powers files
diaad powers reselect
```

`powers analyze`, `powers evaluate`, and `powers rates` read completed coding outputs. They do not rerun the automation pass.

## Automated First-Pass Fields

When automation is enabled and NLP support is available, DIAAD fills:

```text
speech_units
filled_pauses
content_words
num_nouns
tagged_utterance
```

All automated values should be reviewed before analysis or reliability evaluation.

## Fields That Remain Manual

DIAAD does not automatically code:

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
section_e fields
```

Section E fields are generated as sample-level note or descriptor fields. In the current implementation, they are not summarized by `powers analyze`, evaluated by `powers evaluate`, or converted to rates by `powers rates`.

## Read Next

- POWERS module: `docs/manual/04_modules/05_powers/`
- `powers files`: `docs/manual/04_modules/05_powers/05_commands/01_files/`
- `powers reselect`: `docs/manual/04_modules/05_powers/05_commands/05_reselect/`
- Configuration: `docs/manual/02_operation/04_configuration.md`
