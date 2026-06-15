# Digital Conversational Turns Implementation Notes

The Digital Conversational Turns module is implemented under `src/diaad/coding/convo_turns/`.

## File Generation

`turns files` currently passes through the CLI transcript-table prerequisite gate. It builds a base table from the sample sheet in transcript tables, expands rows by configured bins, assigns coder and reliability material, and writes outputs under `coding_templates/`:

- `conversation_turns_template.xlsx`
- `conversation_turns_reliability_template.xlsx`
- `conversation_turns_template_codebook.xlsx`, when configured blinding is active

## Analysis

`turns analyze` recursively reads completed turn workbooks whose filenames match the conversation-turns pattern, parses turn strings, computes speaker-level, group-level, bin-level, session-level, participation, and transition metrics where the required columns are available, and writes one analysis workbook per input file.

Turn strings are parsed digit by digit. Dot markers are counted separately as `mark1` and `mark2` when one or two dots follow a digit. Transition outputs are derived from extracted digit sequences.

## Reliability And Reselection

`turns evaluate` normalizes primary and reliability files, merges rows on sample/session/bin keys, and writes count, sequence, and sample-level reliability sheets plus a report.

`turns reselect` discovers prior turn reliability files and builds replacement reliability rows using shared reselection patterns.

## Boundaries

The implementation is sensitive to turn-string syntax and required columns such as sample identifier, session, bin, and turns. Command pages should give concrete examples and validation guidance.
