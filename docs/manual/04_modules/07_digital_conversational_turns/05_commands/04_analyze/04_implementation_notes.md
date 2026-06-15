# `turns analyze` Implementation Notes

`turns analyze` dispatches to `run_analyze_digital_convo_turns()`.

## Dispatch Path

The main path is:

1. `src/diaad/cli/commands.py` registers `turns analyze`.
2. `src/diaad/cli/dispatch.py` dispatches the command without a transcript-table prerequisite.
3. `src/diaad/core/run_context.py` passes input/output paths and the configured sample identifier.
4. `src/diaad/core/run_wrappers.py` calls `analyze_digital_convo_turns()`.
5. `src/diaad/coding/convo_turns/analysis.py` writes one analysis workbook per matching input workbook.

## File Discovery

The implementation recursively scans the configured input directory for `.xlsx` files whose names match:

```text
.*(Convo|Conversation)_?Turns.*\.xlsx
```

The search does not scan the output directory. It reads the first sheet of each matching workbook.

## Parsing

If `group` is absent and the configured sample identifier is present, the analysis renames the configured sample identifier to `group`.

The parser extracts digits with regular expressions. `extract_turn_counts()` ignores dot markers and returns counts per digit. `extract_turn_stats()` counts turn totals plus `mark1` and `mark2` when one or two dots follow a digit.

Transition sequences are extracted from digits only.

## Workbook Writing

For each input file, the output filename is the input filename with `_analysis` inserted before `.xlsx`. Sheets are written only when the corresponding dataframe is nonempty.

Possible sheets are:

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

## Relevant Sources

- `src/diaad/coding/convo_turns/analysis.py`
- `src/diaad/core/run_context.py`
- `src/diaad/core/run_wrappers.py`
- `tests/test_coding/test_convo_turns/test_analysis.py`
- `tests/test_coding/test_convo_turns/test_identifiers.py`
