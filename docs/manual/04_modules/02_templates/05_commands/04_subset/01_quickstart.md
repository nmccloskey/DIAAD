# `templates subset` Quickstart

`diaad templates subset` creates a randomized sample subset workbook from one input Excel workbook. It is useful for generic reliability, pilot, or protocol-driven sample selection outside DIAAD's specialized coding modules.

## Run

```bash
diaad templates subset --config config
```

## Minimum Inputs

Point `project.input_dir` at a folder containing exactly one `.xlsx` workbook.

That workbook must contain a `samples` sheet with the configured sample identifier column:

```text
sample_id
```

Optionally, the `samples` sheet may also contain:

```text
exclude
```

where values are binary `0` or `1`.

## Primary Output

By default, the command writes:

```text
coding_templates/
  sample_subset.xlsx
```

The workbook contains:

| Sheet | Purpose |
|---|---|
| `samples` | One row per unique sample identifier with `selected` and `excluded`. |
| `subset` | Only rows where `selected == 1`. |

## Immediate Next Step

Inspect both sheets and confirm the selected rows match the intended sampling frame. If you used `exclude`, confirm excluded samples were not selected.

## Read Next

- Templates module quickstart: `docs/manual/04_modules/02_templates/01_quickstart.md`
- Exact file name matching: `docs/manual/03_features/03_exact_file_name_matching.md`
- Configuration: `docs/manual/02_operation/04_configuration.md`
- Generated Example I/O: `docs/manual/03_features/04_generated_example_io.md`
