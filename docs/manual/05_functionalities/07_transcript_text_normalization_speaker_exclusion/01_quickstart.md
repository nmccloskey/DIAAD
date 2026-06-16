# Transcript Text Normalization and Speaker Exclusion Quickstart

DIAAD can normalize transcript text for comparison workflows and exclude configured speaker codes from supported transcript-derived outputs.

## Key Settings

The relevant project settings are:

```yaml
project:
  strip_clan: true
  prefer_correction: true
  lowercase: true
  exclude_speakers: []
```

The defaults are designed for transcript comparison workflows, especially transcription reliability evaluation. Add speaker codes to `exclude_speakers` when a workflow should omit speakers such as an interviewer, clinician, or investigator.

## Common Use

For transcription reliability evaluation, normalization controls how original and reliability transcripts are compared:

- `strip_clan` removes CLAN markup while preserving speech-relevant text;
- `prefer_correction` controls how CLAN correction notation is handled;
- `lowercase` lowercases text before comparison;
- `exclude_speakers` omits selected CHAT participant tiers before comparison.

For transcript-table analysis paths that support speaker exclusion, `exclude_speakers` removes rows whose `speaker` value matches a configured label.

## Read Next

- Configuration: `docs/manual/02_operation/04_configuration.md`
- Transcripts module: `docs/manual/04_modules/01_transcripts/`
- Word Counting commands: `docs/manual/04_modules/04_word_counting/05_commands/`
- Target Vocabulary Coverage commands: `docs/manual/04_modules/06_target_vocabulary_coverage/05_commands/`
