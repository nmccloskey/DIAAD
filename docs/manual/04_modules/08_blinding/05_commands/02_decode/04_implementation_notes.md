# `blinding decode` Implementation Notes

`blinding decode` dispatches to `run_decode_blinding()`.

## Dispatch Path

The main path is:

1. `src/diaad/cli/commands.py` registers `blinding decode`.
2. `src/diaad/cli/dispatch.py` dispatches the command without transcript-table or CHAT prerequisites.
3. `src/diaad/core/run_context.py` passes input/output paths and advanced blinding configuration.
4. `src/diaad/core/run_wrappers.py` calls `decode_blinding()`.
5. `src/diaad/blinding/decode.py` writes the decoded workbook.

## Discovery

If `advanced.codebook_filename` is configured, the command resolves that filename in the input directory. Otherwise, it searches recursively for `.xlsx` files whose stem contains `blind_codebook`.

The target workbook is one `.xlsx` file that is not the selected codebook, does not contain `blind_codebook` in its stem, and is not an Excel temporary file.

## Decoding

The command validates the codebook with `validate_decode_codebook()`, then calls `unblind_dataframe()` with:

```text
strict = false
suffix = _blinded
```

`unblind_dataframe()` first looks for suffixed analysis-style columns such as `sample_id_blinded`. If a suffixed column is absent but the raw column name is present, it treats the raw column as in-place blind codes and decodes those values.

## Output Naming

Outputs are written under:

```text
blinding/
```

For a target workbook named `analysis_blinded.xlsx`, the output name is:

```text
analysis_blinded_decoded.xlsx
```

## Relevant Sources

- `src/diaad/blinding/decode.py`
- `src/diaad/metadata/unblinding.py`
- `src/diaad/core/run_context.py`
- `src/diaad/core/config.py`
- `tests/test_blinding/test_commands.py`
- `tests/test_metadata/test_unblinding.py`
