# Digital Conversational Turns Implementation Notes

The Digital Conversational Turns module is implemented under `src/diaad/coding/convo_turns/`.

## Analysis

`turns analyze` reads the exact workbook named by `advanced.dct_coding_filename`, parses turn strings, computes speaker-level, group-level, bin-level, session-level, participation, and transition metrics where the required columns are available, and writes one analysis workbook.

Turn strings are parsed digit by digit. Dot markers are counted separately as `mark1` and `mark2` when one or two dots follow a digit. Transition outputs are derived from extracted digit sequences.

## Reliability

`turns evaluate` normalizes primary and reliability files, merges rows on sample/session/bin keys, and writes count, sequence, and sample-level reliability sheets plus a report.

## Boundaries

The implementation is sensitive to turn-string syntax and required columns such as sample identifier, session, bin, and turns. Command pages should give concrete examples and validation guidance.
