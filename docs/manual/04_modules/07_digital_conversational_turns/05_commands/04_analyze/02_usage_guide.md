# `turns analyze` Usage Guide

Use `diaad turns analyze` when DCT strings are complete and ready to become quantitative summaries.

## File Discovery

The command searches the configured input and output directories recursively for exactly one primary DCT coding workbook. By default, that workbook is:

```text
conversation_turns.xlsx
```

The filename is controlled by `advanced.dct_coding_filename`. If no matching file is found, or more than one matching file is found, DIAAD stops with a discovery error.

Until the transcript-table fallback pass is implemented, `turns analyze` analyzes only the primary manual coding workbook.

## Required Fields

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

## Turn-String Interpretation

Each digit is counted as one turn for that speaker. Dots are interpreted only when they immediately follow a digit:

```text
1.
1..
```

One dot contributes to `mark1`; two dots contribute to `mark2`. The transition sequence is extracted from digits only.

The parser is digit-by-digit. A string containing `10` is interpreted as speaker `1` followed by speaker `0`.

## Output Sheets

`speaker_level_turns` and `group_level_summary` summarize turn and marker counts.

`bin_level_turns` appears when `bin` is present. `session_level_summary` and `participation_level_turns` appear when `session` is present.

`speaker_level_ratios` summarizes transition probabilities into participant-to-participant, participant-to-clinician, and clinician-to-participant categories.

`speaker_matrix_<group>` sheets contain transition matrices for each group. Excel sheet names are truncated to fit Excel's sheet-name limit.

## Common Problems

If no analysis workbook appears, check the input filename and the required columns.

If transition metrics look strange, inspect whether `0` consistently represents the clinician/non-client category and whether participant digits were used consistently across samples.

If counts are unexpectedly high, check whether multi-digit identifiers such as `10` were entered; those are parsed as two separate speakers.

## Read Next

- `turns evaluate` research context: `docs/manual/04_modules/07_digital_conversational_turns/05_commands/02_evaluate/03_research_context.md`
- Digital Conversational Turns implementation notes: `docs/manual/04_modules/07_digital_conversational_turns/04_implementation_notes.md`
- Exact file-name matching: `docs/manual/03_features/03_exact_file_name_matching.md`
