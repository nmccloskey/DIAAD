# `cus reselect` Implementation Notes

`cus reselect` dispatches to `reselect_cu_rel()`.

## Dispatch Path

The main path is:

1. `src/diaad/cli/commands.py` registers `cus reselect`.
2. `src/diaad/cli/dispatch.py` dispatches it as a CU command.
3. `src/diaad/core/run_context.py` threads metadata fields, reliability fraction, random seed, and sample identifier column.
4. `src/diaad/core/run_wrappers.py` calls `reselect_cu_rel()`.
5. `src/diaad/coding/compl_utts/rel_reselection.py` writes reselected reliability workbooks.

## Pair Discovery

Shared reselection utilities discover original and reliability workbooks with:

```text
*cu_coding.xlsx
*cu_reliability_coding.xlsx
```

Files are matched by metadata-value tuples derived from configured metadata fields. Without metadata fields, file stems are used as fallback keys.

## Used Sample Detection

Used sample IDs are collected from matched prior reliability workbooks. Sample IDs are normalized to stripped strings before comparison.

New sample IDs are selected from original coding rows whose sample IDs are not already used.

## Output Naming

Outputs are written under:

```text
reselected_cu_coding_reliability/
```

For the usual source file `cu_coding.xlsx`, the output is:

```text
reselected_cu_reliability_coding.xlsx
```

## Relevant Sources

- `src/diaad/coding/compl_utts/rel_reselection.py`
- `src/diaad/coding/utils/reselection_utils.py`
- `src/diaad/coding/utils/sampling.py`
