# `turns analyze` Implementation Notes

`turns analyze` dispatches to `run_analyze_digital_convo_turns()`.

## Dispatch Path

The main path is:

1. `src/diaad/cli/commands.py` registers `turns analyze`.
2. `src/diaad/cli/dispatch.py` dispatches the command without a transcript-table prerequisite.
3. `src/diaad/core/run_context.py` passes input/output paths, excluded speakers, the configured sample identifier, `advanced.dct_coding_filename`, and `advanced.transcript_table_filename`.
4. `src/diaad/core/run_wrappers.py` calls `analyze_digital_convo_turns()` for the DCT workbook.
5. If the DCT workbook is absent, the wrapper calls `ctx.ensure_transcript_tables()` and reruns `analyze_digital_convo_turns()` with transcript-table mode enabled.
6. `src/diaad/coding/convo_turns/analysis.py` writes one analysis workbook for the selected source.

## File Discovery

The implementation uses DIAAD's exact file discovery policy to find one workbook named by:

```text
advanced.dct_coding_filename
```

The search scans the configured input and output directories. It falls back only when no DCT workbook is found. Multiple DCT matches are treated as duplicates and do not fall back.

Fallback uses the same exact-file policy for:

```text
advanced.transcript_table_filename
```

Multiple transcript table matches are treated as duplicates. Missing transcript tables are handled by `ctx.ensure_transcript_tables()`, so `project.auto_tabularize` decides whether CHAT files may be tabularized automatically.

## Parsing

Analysis normalizes both sources into event rows with:

```text
sample_id
session
bin
speaker
sequence_position
mark1
mark2
source
```

`session` and `bin` are optional. Transcript-derived rows do not synthesize bins.

DCT parsing extracts digit-speaker events with regular expressions. `extract_turn_counts()` ignores dot markers and returns counts per digit. `extract_turn_stats()` counts turn totals plus `mark1` and `mark2` when one or two dots follow a digit.

Transcript-table parsing reads joined transcript rows and treats each speaker tag as a single sequence token.

Speakers listed in `project.exclude_speakers` are pooled into the disinterest category rather than dropped. When the list is not empty, its first value is the category label and DCT `0` is mapped to that label.

Transition sequences are derived from normalized speaker tokens.

## Workbook Writing

DCT output filenames insert `_analysis` before `.xlsx`. Transcript-table fallback inserts `_turns_analysis` before `.xlsx`. Sheets are written only when the corresponding dataframe is nonempty.

Possible sheets are:

```text
bin_level_turns
participation_level_turns
session_level_summary
speaker_level_turns
group_level_summary
summary_statistics
speaker_level_ratios
speaker_label_mapping
speaker_matrix_<group>
```

The `speaker_label_mapping` sheet is built from normalized event rows and logs the original speaker label, mapped speaker label, mapping reason, and source. The same mapping is also summarized in the run log.

## Relevant Sources

- `src/diaad/coding/convo_turns/analysis.py`
- `src/diaad/core/run_context.py`
- `src/diaad/core/run_wrappers.py`
- `tests/test_coding/test_convo_turns/test_analysis.py`
- `tests/test_coding/test_convo_turns/test_identifiers.py`
