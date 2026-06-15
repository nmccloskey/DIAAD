# `turns files` Implementation Notes

`turns files` dispatches to `run_make_digital_convo_turn_files()`.

## Dispatch Path

The main path is:

1. `src/diaad/cli/commands.py` registers `turns files`.
2. `src/diaad/cli/dispatch.py` marks `turns files` as requiring transcript tables before dispatch.
3. `src/diaad/core/run_context.py` passes input/output paths, reliability fraction, bin count, coder count, blinding configuration, random seed, sample identifier, and transcript-table filename.
4. `src/diaad/core/run_wrappers.py` calls `make_digital_convo_turn_files()`.
5. `src/diaad/coding/convo_turns/files.py` writes the template exports.

## Template Construction

The implementation extracts the sample table from the transcript table workbook, requires the configured sample identifier column, drops duplicate sample identifiers, and creates blank `session` and `turns` fields.

Rows are expanded once per configured bin label. The default bin labels come from `project.num_bins` and are simple numeric labels beginning at `1`.

Coder assignment and reliability-subset construction use shared template helpers from `src/diaad/coding/templates/utils.py`.

## Outputs

The outputs are written under the shared template subdirectory:

```text
coding_templates/
  conversation_turns_template.xlsx
  conversation_turns_reliability_template.xlsx
```

The optional codebook is written as `conversation_turns_template_codebook.xlsx` only when configured coding blinding is active.

## Relevant Sources

- `src/diaad/coding/convo_turns/files.py`
- `src/diaad/coding/templates/utils.py`
- `src/diaad/core/run_context.py`
- `src/diaad/core/run_wrappers.py`
- `src/diaad/cli/dispatch.py`
- `tests/test_coding/test_convo_turns/test_identifiers.py`
