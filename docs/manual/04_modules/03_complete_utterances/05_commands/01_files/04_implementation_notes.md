# `cus files` Implementation Notes

`cus files` dispatches to `make_cu_coding_files()`.

## Dispatch Path

The main path is:

1. `src/diaad/cli/commands.py` registers `cus files`.
2. `src/diaad/cli/dispatch.py` marks it as requiring transcript tables.
3. `src/diaad/core/run_context.py` threads CU settings, identifier columns, excluded speakers, blinding config, and transcript table filename.
4. `src/diaad/core/run_wrappers.py` calls `make_cu_coding_files()`.
5. `src/diaad/coding/compl_utts/files.py` writes the workbooks.

## Transcript Input

The command discovers the configured transcript table filename in the input directory or current run output directory. It extracts joined transcript data, shuffles sample blocks, and drops transcript-table columns that are not needed for CU coding.

The stimulus column is preserved when it is explicitly configured or found through the legacy stimulus-column fallback.

## Output Files

The command writes under:

```text
cu_coding/
```

Current filenames are:

```text
cu_coding.xlsx
cu_reliability_coding.xlsx
cu_blind_codebook.xlsx
```

The blind codebook is only written when blinding is active and produces codebook rows.

## Reliability Selection

Reliability sample count is calculated with the shared `calc_subset_size()` helper. Selection preserves sample blocks: when a sample is selected, all rows for that sample are included in the reliability workbook.

## Relevant Sources

- `src/diaad/coding/compl_utts/files.py`
- `src/diaad/coding/utils/sampling.py`
- `src/diaad/coding/utils/coders.py`
- `src/diaad/coding/utils/transcript.py`
- `tests/test_coding/test_compl_utts/test_identifiers.py`
