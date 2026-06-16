# Transcript Text Normalization and Speaker Exclusion Implementation Notes

Normalization and speaker exclusion are implemented in both transcript reliability utilities and shared transcript-row helpers.

## Source Anchors

Primary sources:

- `src/diaad/transcripts/transcription_reliability_evaluation.py`
- `src/diaad/coding/utils/transcript.py`
- `src/diaad/core/config.py`
- `src/diaad/core/run_context.py`

Relevant tests:

- `tests/test_coding/test_utils/test_transcript.py`
- transcript reliability tests under `tests/test_transcripts/`
- module-specific analysis tests that pass `exclude_speakers`

## Reliability Text Pipeline

`process_utterances()` in `transcription_reliability_evaluation.py` applies this sequence:

1. Extract utterance text from a `pylangacq.Reader`, or accept a plain string unchanged.
2. Exclude configured CHAT participant tiers during extraction when the source is a Reader.
3. Process CLAN correction notation using `prefer_correction`.
4. Strip CLAN markup when `strip_clan` is true.
5. Collapse whitespace.
6. Lowercase when `lowercase` is true.

The reliability evaluation command receives `exclude_speakers`, `strip_clan`, `prefer_correction`, and `lowercase` from `RunContext`.

## Transcript-Row Exclusion

`drop_excluded_speaker_rows()` in `src/diaad/coding/utils/transcript.py` is the shared helper for transcript-table DataFrames. It returns the input unchanged when no speakers are configured or when no `speaker` column is present.

When filtering applies, it trims and lowercases both configured labels and DataFrame speaker values, removes matching rows, logs the number excluded, and returns a copy.

This helper is used by transcript-derived analysis paths such as Word Counting and Target Vocabulary Coverage. Module command pages describe whether a specific command applies the filter during file generation, analysis, or both.

## Boundary With CHAT Export

`transcripts chats` performs limited text regularization for reconstructed CHAT output, such as punctuation normalization and safe file-name handling. That is separate from the reliability normalization settings described here.

## Read Next

- Configuration implementation notes: `docs/manual/05_functionalities/01_configuration_sources_defaults_overrides/04_implementation_notes.md`
- Transcript preprocessing implementation notes: `docs/manual/05_functionalities/06_transcript_preprocessing_tabularization_chat_export/04_implementation_notes.md`
- Testing: `docs/manual/02_operation/05_testing.md`
