# `templates samples` Implementation Notes

`templates samples` dispatches to `make_sample_template_files()`.

## Dispatch Path

The main path is:

1. `src/diaad/cli/commands.py` registers `templates samples`.
2. `src/diaad/cli/dispatch.py` marks it as requiring transcript tables.
3. `src/diaad/core/run_context.py` threads `num_bins`, coder settings, identifier columns, blinding config, and transcript table filename.
4. `src/diaad/core/run_wrappers.py` calls `make_sample_template_files()`.
5. `src/diaad/coding/templates/samples.py` builds the primary and reliability workbooks.

## Data Preparation

The command reads the transcript table `samples` sheet, keeps one row per configured sample identifier, optionally carries a configured stimulus column, then expands each sample by bin label.

Bin labels are canonical integers from `1` through `project.num_bins`. `project.num_bins` must be at least `1`.

## Output Files

The implementation writes under:

```text
coding_templates/
```

Output filenames are currently fixed by the Templates module:

```text
sample_coding_template.xlsx
sample_reliability_template.xlsx
sample_template_codebook.xlsx
```

The workbook sheet name is `coding_template`.

## Reliability And Blinding

The reliability subset is built from sample identifiers after bin expansion and coder assignment. The selected subset keeps all rows for selected samples.

When coding blinding is enabled, the command blinds configured columns in both the primary and reliability workbook using the same generated codebook.

## Relevant Sources

- `src/diaad/coding/templates/samples.py`
- `src/diaad/coding/templates/utils.py`
- `src/diaad/coding/utils/coders.py`
- `src/diaad/coding/utils/sampling.py`
- `tests/test_coding/test_templates/test_utils.py`
- `tests/test_coding/test_templates/test_identifiers.py`
