# `templates subset` Implementation Notes

`templates subset` dispatches to `make_sample_subset_file()`.

## Dispatch Path

The main path is:

1. `src/diaad/cli/commands.py` registers `templates subset`.
2. `src/diaad/cli/dispatch.py` dispatches it without requiring transcript tables.
3. `src/diaad/core/run_context.py` threads input/output paths, reliability fraction, sample identifier column, and random seed.
4. `src/diaad/core/run_wrappers.py` calls `make_sample_subset_file()`.
5. `src/diaad/coding/templates/subset.py` reads one workbook and writes `sample_subset.xlsx`.

## Input Discovery

The implementation uses `find_one_file_by_extension()` with `.xlsx`. This differs from exact filename discovery: the filename can vary, but the active input directory must resolve to exactly one `.xlsx` workbook.

Temporary Excel files beginning with `~$` are ignored by discovery.

## Validation

The input workbook must contain a `samples` sheet. That sheet must contain the configured sample identifier column and no blank sample identifiers.

If an `exclude` column exists, values are coerced to binary integers. Accepted values include `0`, `1`, booleans, and numeric equivalents. Other values raise an error.

## Output

The command writes:

```text
coding_templates/sample_subset.xlsx
```

with sheets:

```text
samples
subset
```

The output `samples` sheet has one row per unique sample identifier plus `selected` and `excluded`. The `subset` sheet contains selected rows only.

## Selection Logic

Subset size is calculated with the shared `calc_subset_size()` helper. Selection uses `random.Random(seed)` from the configured `project.random_seed`.

When `exclude` is present, target size is still based on the full sample set, while sampling is limited to non-excluded candidates.

## Relevant Sources

- `src/diaad/coding/templates/subset.py`
- `src/diaad/metadata/discovery.py`
- `src/diaad/coding/utils/sampling.py`
- `tests/test_coding/test_templates/test_subset.py`
