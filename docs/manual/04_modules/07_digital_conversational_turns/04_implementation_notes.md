# Digital Conversational Turns Implementation Notes

The Digital Conversational Turns module is implemented under `src/diaad/coding/convo_turns/`.

## File Generation

`turns files` builds a base table from transcript tables, expands rows by configured bins, assigns coder and reliability material, and writes outputs under `coding_templates/`:

- `conversation_turns_template.xlsx`
- `conversation_turns_reliability_template.xlsx`
- `conversation_turns_template_codebook.xlsx`

## Analysis

`turns analyze` reads completed turn workbooks, parses turn strings, computes speaker-level, group-level, bin-level, session-level, participation, and transition metrics where the required columns are available, and writes an analysis workbook.

Transition outputs are derived from extracted speaker sequences.

## Reliability And Reselection

`turns evaluate` normalizes primary and reliability files, merges rows on sample/session/bin keys, and writes count, sequence, and sample-level reliability sheets plus a report.

`turns reselect` discovers prior turn reliability files and builds replacement reliability rows using shared reselection patterns.

## Boundaries

The implementation is sensitive to turn-string syntax and required columns such as sample identifier, session, bin, and turns. Command pages should give concrete examples and validation guidance.
