# `turns analyze` Quickstart

`diaad turns analyze` summarizes completed DCT turn strings across speakers, bins, sessions, groups, and transitions.

## Run

```bash
diaad turns analyze --config config
```

## Minimum Inputs

Place completed DCT workbooks under the configured input directory. The command scans recursively for Excel filenames matching the conversation-turns pattern, such as:

```text
conversation_turns_template.xlsx
```

Each workbook must include `turns` and either the configured sample identifier column or `group`.

## Primary Output

For each matching input workbook, DIAAD writes an analysis workbook in the current output directory. For the standard template filename, the output is:

```text
conversation_turns_template_analysis.xlsx
```

Possible sheets include:

```text
bin_level_turns
participation_level_turns
session_level_summary
speaker_level_turns
group_level_summary
summary_statistics
speaker_level_ratios
speaker_matrix_<group>
```

## Immediate Next Step

Inspect speaker and group summaries first, then use bin/session and transition sheets only after confirming that the input rows used consistent session, bin, and speaker-code conventions.

## Read Next

- `turns analyze` usage guide: `docs/manual/04_modules/07_digital_conversational_turns/05_commands/04_analyze/02_usage_guide.md`
- `turns analyze` research context: `docs/manual/04_modules/07_digital_conversational_turns/05_commands/04_analyze/03_research_context.md`
- Digital Conversational Turns research context: `docs/manual/04_modules/07_digital_conversational_turns/03_research_context.md`
- Generated Example I/O: `docs/manual/03_features/04_generated_example_io.md`
