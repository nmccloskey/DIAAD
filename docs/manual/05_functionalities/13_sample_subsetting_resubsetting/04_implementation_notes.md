# General Sample Subsetting and Re-Subsetting Implementation Notes

General subsetting is implemented by the Templates module with shared sampling helpers.

## Source Anchors

Primary sources:

- `src/diaad/coding/templates/subset.py`
- `src/diaad/coding/utils/sampling.py`
- `src/diaad/core/run_context.py`
- `src/diaad/metadata/discovery.py`

Relevant tests:

- `tests/test_coding/test_templates/test_subset.py`
- `tests/test_coding/test_utils/test_sampling.py`

## Input Discovery

`make_sample_subset_file()` calls `find_one_file_by_extension()` and requires exactly one `.xlsx` file in the input directory. Temporary Excel lock files beginning with `~$` are ignored by the discovery helper.

The workbook must contain a sheet named `samples`. That sheet must contain the configured sample identifier column.

## Exclusion Handling

If the `samples` sheet has no `exclude` column, all unique samples are eligible.

If an `exclude` column is present, values must be binary `0` or `1`. Boolean values and numeric `0.0` or `1.0` are also accepted. Other values raise an error.

When duplicate sample IDs have mixed exclusion values, DIAAD logs a warning and treats the sample as excluded if any row is excluded.

## Selection

`_select_samples()` computes the target size with `calc_subset_size()`, samples from eligible candidates using the configured random seed, and caps the draw when there are fewer eligible candidates than the target count.

The output frame contains:

```text
sample_id
selected
excluded
```

using the configured sample identifier column name.

## Output

The command writes:

```text
coding_templates/sample_subset.xlsx
```

with:

- `samples`: all sample statuses;
- `subset`: selected rows only.

## Read Next

- Reliability implementation notes: `docs/manual/05_functionalities/12_reliability_selection_evaluation_reselection/04_implementation_notes.md`
- File discovery implementation notes: `docs/manual/05_functionalities/09_configured_filenames_file_discovery_input_selection/04_implementation_notes.md`
- Templates implementation notes: `docs/manual/04_modules/02_templates/04_implementation_notes.md`
