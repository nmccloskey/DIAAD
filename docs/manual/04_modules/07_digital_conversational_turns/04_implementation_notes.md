# Digital Conversational Turns Implementation Notes

The Digital Conversational Turns module is implemented under `src/diaad/coding/convo_turns/`.

## Analysis

`turns analyze` first reads the exact workbook named by `advanced.dct_coding_filename`. If that workbook is absent, the wrapper checks for or creates transcript tables according to `project.auto_tabularize`, then analyzes the exact configured transcript table instead.

Both sources are normalized into an internal event table with sample, optional session/bin, speaker, sequence position, marker counts, and source. DCT strings become digit-speaker events with dot markers. Transcript tables become token-speaker events from ordered utterance rows.

Speakers listed in `project.exclude_speakers` are pooled into the non-client/disinterest category rather than dropped. If any excluded speakers are configured, the first configured value replaces the DCT `0` category for ratio and transition summaries.

Turn strings are parsed digit by digit. Dot markers are counted separately as `mark1` and `mark2` when one or two dots follow a digit. Transcript-derived input does not synthesize bins, so bin-level sheets are only available from manual DCT workbooks that contain `bin`.

## Reliability

`turns evaluate` normalizes primary and reliability files, merges rows on sample/session/bin keys, and writes count, sequence, and sample-level reliability sheets plus a report.

## Boundaries

The implementation is sensitive to turn-string syntax and required columns such as sample identifier, session, bin, and turns. Command pages should give concrete examples and validation guidance.
