# `turns analyze` Usage Guide

Use `diaad turns analyze` when DCT strings or transcript-table speaker sequences are ready to become quantitative summaries.

## File Discovery

The command first searches the configured input and output directories recursively for exactly one primary DCT coding workbook. By default, that workbook is:

```text
conversation_turns.xlsx
```

The filename is controlled by `advanced.dct_coding_filename`. If more than one matching DCT workbook is found, DIAAD stops with a discovery error.

If no DCT workbook is found, DIAAD falls back to the exact configured transcript table named by `advanced.transcript_table_filename`. Duplicate transcript tables also stop the command with a discovery error.

The fallback respects `project.auto_tabularize`. If transcript tables are absent and `auto_tabularize: true`, DIAAD creates transcript tables from CHAT files before analysis. If transcript tables are absent and `auto_tabularize: false`, the command stops and asks for transcript tables or CHAT tabularization.

## Manual DCT Required Fields

The input workbook must include:

```text
turns
```

It must also include either:

```text
sample_id
group
```

`sample_id` may be replaced by the configured `advanced.sample_id_column`. When the configured sample identifier is present and `group` is absent, DIAAD renames it internally to `group`.

`session` and `bin` are optional, but they unlock additional output sheets and metrics.

## Transcript-Table Required Fields

Transcript fallback reads the `samples` and `utterances` sheets from the configured transcript table. It requires the configured sample identifier column and `speaker` in the joined utterance data. Utterance order comes from `position`, `position_sub`, and row order when those columns are available.

Transcript-derived input uses speaker tags directly and does not synthesize bins, so `bin_level_turns` is not produced from transcript fallback.

## Turn-String Interpretation

Each digit is counted as one turn for that speaker. Dots are interpreted only when they immediately follow a digit:

```text
1.
1..
```

One dot contributes to `mark1`; two dots contribute to `mark2`. Manual DCT strings still parse speakers digit by digit.

The parser is digit-by-digit. A string containing `10` is interpreted as speaker `1` followed by speaker `0`.

Transcript-table fallback does not parse speaker tags as characters. Each speaker tag is treated as one sequence token, so tags such as `INV`, `CHI`, or `MOT` remain intact.

Speakers listed in `project.exclude_speakers` are pooled into the DCT non-client category rather than dropped. If any excluded speakers are configured, the first listed speaker becomes the category label. For example, `exclude_speakers: [INV, INV2]` maps both `INV` and `INV2` to `INV`, and DCT `0` is also reported as `INV`.

## Output Sheets

`speaker_level_turns` and `group_level_summary` summarize turn and marker counts.

`bin_level_turns` appears when `bin` is present. `session_level_summary` and `participation_level_turns` appear when `session` is present.

`speaker_level_ratios` summarizes transition probabilities into participant-to-participant, participant-to-clinician, and clinician-to-participant categories.

`speaker_label_mapping` records original speaker labels, mapped speaker labels, and the reason for each mapping. This is especially useful when `project.exclude_speakers` pools multiple transcript speaker tags into the DCT non-client category.

`speaker_matrix_<group>` sheets contain transition matrices for each group. Excel sheet names are truncated to fit Excel's sheet-name limit.

## Common Problems

If no analysis workbook appears, check the DCT filename, transcript table filename, `auto_tabularize` setting, and required columns.

If transition metrics look strange, inspect whether `0` or the first configured excluded speaker consistently represents the clinician/non-client category and whether participant speakers were used consistently across samples.

If counts are unexpectedly high, check whether multi-digit identifiers such as `10` were entered; those are parsed as two separate speakers.

## Read Next

- `turns evaluate` research context: `docs/manual/04_modules/07_digital_conversational_turns/05_commands/02_evaluate/03_research_context.md`
- Digital Conversational Turns implementation notes: `docs/manual/04_modules/07_digital_conversational_turns/04_implementation_notes.md`
- Exact file-name matching: `docs/manual/03_features/03_exact_file_name_matching.md`
