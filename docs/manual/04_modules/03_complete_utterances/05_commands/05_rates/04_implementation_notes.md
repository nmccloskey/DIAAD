# `cus rates` Implementation Notes

`cus rates` dispatches to `calculate_cu_rates()`.

## Dispatch Path

The main path is:

1. `src/diaad/cli/commands.py` registers `cus rates`.
2. `src/diaad/cli/dispatch.py` dispatches it as a CU command.
3. `src/diaad/core/run_context.py` threads configured CU sample-summary and speaking-time filenames.
4. `src/diaad/core/run_wrappers.py` calls `calculate_cu_rates()`.
5. `src/diaad/coding/compl_utts/rates.py` writes `cu_coding_rates.xlsx`.

## Input Discovery

The command uses exact filename discovery for the configured CU sample summary and speaking-time workbook. Defaults are:

```text
cu_coding_by_sample_long.xlsx
speaking_times.xlsx
```

Both are searched in the input directory and current run output directory.

## Rate Calculation

The command reads the long CU sample summary, reads and standardizes the speaking-time workbook, merges by the configured sample identifier, and adds per-minute columns for:

```text
cu
p_sv
p_rel
```

The shared rate utility computes each rate as numerator divided by `speaking_minutes`. `speaking_minutes` is calculated as `speaking_time / 60`.

## Output File

The command writes:

```text
cu_coding_analysis/cu_coding_rates.xlsx
```

The output filename is fixed by the CU rates implementation.

## Relevant Sources

- `src/diaad/coding/compl_utts/rates.py`
- `src/diaad/coding/utils/rates.py`
- `tests/test_coding/test_compl_utts/test_identifiers.py`
