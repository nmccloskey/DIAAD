# `blinding encode` Implementation Notes

`blinding encode` dispatches to `run_encode_blinding()`.

## Dispatch Path

The main path is:

1. `src/diaad/cli/commands.py` registers `blinding encode`.
2. `src/diaad/cli/dispatch.py` dispatches the command without transcript-table or CHAT prerequisites.
3. `src/diaad/core/run_context.py` passes input/output paths, advanced blinding configuration, and random seed.
4. `src/diaad/core/run_wrappers.py` calls `encode_blinding()`.
5. `src/diaad/blinding/encode.py` writes the blinded workbook, diagnostics workbook, and codebook.

## Discovery

If `advanced.codebook_filename` is configured, the command resolves that filename in the input directory. Otherwise, it searches recursively for `.xlsx` files whose stem contains `blind_codebook`.

The target workbook is one `.xlsx` file that is not the selected codebook, does not contain `blind_codebook` in its stem, and is not an Excel temporary file.

## Encoding

When no codebook exists, the command uses `AdvancedConfig.get_blind_cols("analysis")`, which currently returns `advanced.blind_columns`. Missing requested columns are skipped; at least one requested column must be present.

When a codebook exists, the command reads target columns from the codebook's `column` field and validates value coverage before encoding.

Encoding uses `blind_analysis_dataframe()`, which appends suffixed blinded columns, drops the raw target columns from the blinded output, and writes diagnostics.

## Output Naming

Outputs are written under:

```text
blinding/
```

For a target workbook named `analysis.xlsx`, the output names are:

```text
analysis_blinded.xlsx
analysis_blinding_diagnostics.xlsx
blind_codebook.xlsx
```

## Relevant Sources

- `src/diaad/blinding/encode.py`
- `src/diaad/metadata/blinding.py`
- `src/diaad/core/run_context.py`
- `src/diaad/core/config.py`
- `tests/test_blinding/test_commands.py`
- `tests/test_metadata/test_blinding.py`
