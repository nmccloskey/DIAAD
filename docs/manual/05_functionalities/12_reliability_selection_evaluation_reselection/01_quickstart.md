# Reliability Selection, Evaluation, and Reselection Quickstart

DIAAD supports reliability workflows for transcription and several manual coding modules. The shared pattern is: select or generate reliability material, complete an independent reliability pass, evaluate agreement, and reselect only when another round is needed.

## Main Reliability Commands

| Area | Commands |
|---|---|
| Transcription reliability | `diaad transcripts select`; `diaad transcripts evaluate`; `diaad transcripts reselect` |
| Complete Utterances reliability | `diaad cus evaluate`; `diaad cus reselect` |
| Word Counting reliability | `diaad words evaluate`; `diaad words reselect` |
| POWERS reliability | `diaad powers evaluate`; `diaad powers reselect` |
| Digital Conversational Turns reliability | `diaad turns evaluate` |
| Target Vocabulary Coverage validation | `diaad vocab check` |

`vocab check` is not a manual-coder reliability command. It validates target-vocabulary resources and should be read with the Target Vocabulary Coverage module pages.

## Typical Path

For coding modules with generated reliability workbooks:

1. Generate the primary and reliability coding files.
2. Complete primary coding and independent reliability coding.
3. Run the module-specific `evaluate` command.
4. Inspect the report and detailed workbook.
5. Use `reselect` only if the original reliability round does not meet the project's criteria or more material is required.

For transcription reliability, use `transcripts select` before the independent reliability transcription round, then `transcripts evaluate` after both transcript sets exist.

## Key Setting

Reliability selection and reselection use:

```yaml
project:
  reliability_fraction: 0.2
```

This fraction determines the target subset size for many reliability selection paths.

## Read Next

- Transcripts reliability commands: `docs/manual/04_modules/01_transcripts/05_commands/`
- Complete Utterances commands: `docs/manual/04_modules/03_complete_utterances/05_commands/`
- Word Counting commands: `docs/manual/04_modules/04_word_counting/05_commands/`
- POWERS commands: `docs/manual/04_modules/05_powers/05_commands/`
- Digital Conversational Turns commands: `docs/manual/04_modules/07_digital_conversational_turns/05_commands/`
