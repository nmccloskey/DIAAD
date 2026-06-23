# Coder Identifier Columns Implementation Notes

Coder identifier handling is spread across the coding modules because each module has a different workbook shape. The shared intent is consistent: generated files expose coder assignment columns, while analysis commands do not require those columns.

## Source Anchors

Primary sources:

- `src/diaad/core/config.py`
- `src/diaad/core/run_context.py`
- `src/diaad/coding/utils/coders.py`
- `src/diaad/coding/templates/utils.py`
- `src/diaad/coding/templates/samples.py`
- `src/diaad/coding/templates/utterances.py`
- `src/diaad/coding/compl_utts/files.py`
- `src/diaad/coding/compl_utts/analysis.py`
- `src/diaad/coding/compl_utts/rel_reselection.py`
- `src/diaad/coding/word_counts/files.py`
- `src/diaad/coding/word_counts/analysis.py`
- `src/diaad/coding/powers/files.py`
- `src/diaad/coding/powers/analysis.py`
- `src/diaad/coding/powers/rel_reselection.py`

Relevant tests:

- `tests/test_coding/test_templates/test_utils.py`
- `tests/test_coding/test_templates/test_identifiers.py`
- `tests/test_coding/test_compl_utts/test_identifiers.py`
- `tests/test_coding/test_word_counts/test_files.py`
- `tests/test_coding/test_word_counts/test_identifiers.py`
- `tests/test_coding/test_powers/test_identifiers.py`

## Configuration Path

`project.num_coders` is parsed as an integer in the project configuration. The packaged default is:

```yaml
project:
  num_coders: 0
```

`RunContext` passes this value into template, CU, word-count, and POWERS file-generation functions through command-specific keyword builders.

## Shared Template Helpers

`diaad.coding.templates.utils.resolve_template_coder_ids()` converts `num_coders` into string labels:

```text
0 -> [""]
2 -> ["1", "2"]
```

`assign_template_coders()` assigns primary coder IDs by sample. `build_reliability_subset()` selects reliability samples and, when multiple coder labels are available, assigns the reliability rows to the alternate role from the generated coder assignment tuple.

These helpers are used by the general template commands.

## Complete Utterances

CU file generation uses integer coder labels and resolves `num_coders` into a mode:

| Mode | Trigger | User-facing columns |
|---|---|---|
| `zero` | `num_coders <= 0` | `coder_id`, blank values |
| `single` | `num_coders == 1` | `coder_id`, value `1` |
| `two` | `num_coders == 2` | `coder_id`, primary/reliability assignments |
| `three` | `num_coders >= 3` | `coder1_id`, `coder2_id`, `coder3_id` with `c1_`, `c2_`, `c3_` coding fields |

If `num_coders` is greater than `3`, CU logs a warning and uses only coder IDs `1`, `2`, and `3`.

CU analysis drops coder administrative columns before summary. The drop list includes current names and compatibility names, including `coder_id`, `coder1_id`, `coder2_id`, `coder3_id`, legacy `id`, and older draft names such as `c1_id` and `c1_coder_id`.

## Word Counting

Word Counting file generation writes a `coder_id` column in primary and reliability workbooks. It uses integer labels `1..num_coders` when coders are configured, and blank values when `num_coders` is `0`.

Word-count analysis drops administrative columns when present, including current `coder_id` and legacy `id`, then summarizes the configured word-count field.

## POWERS

POWERS file generation writes `coder_id` in the utterance-level coding sheet. The helper `_resolve_powers_coder_ids()` returns blank for `num_coders <= 0` and string labels `1..num_coders` otherwise.

POWERS reselection ensures `coder_id` exists for reliability-side administrative shape and clears it, along with manual POWERS fields, so newly selected rows are ready for fresh reliability coding.

POWERS analysis groups and summarizes completed POWERS coding fields. It does not require `coder_id`.

## Naming Update

Current outputs and documentation use `coder_id` instead of the older generic `id` name. CU's three-coder schema uses `coder1_id`, `coder2_id`, and `coder3_id`; it does not use `c1_coder_id`, `c2_coder_id`, or `c3_coder_id`.

Analysis cleanup remains tolerant of legacy `id` columns so older workbooks can be handled where possible.

## Read Next

- Configuration implementation notes: `docs/manual/05_functionalities/01_configuration_sources_defaults_overrides/04_implementation_notes.md`
- Reliability implementation notes: `docs/manual/05_functionalities/12_reliability_selection_evaluation_reselection/04_implementation_notes.md`
- Configurable identifiers implementation notes: `docs/manual/05_functionalities/10_configurable_sample_utterance_identifiers/04_implementation_notes.md`
