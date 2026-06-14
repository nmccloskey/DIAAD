# `cus analyze` Implementation Notes

`cus analyze` dispatches to `analyze_cu_coding()`.

## Dispatch Path

The main path is:

1. `src/diaad/cli/commands.py` registers `cus analyze`.
2. `src/diaad/cli/dispatch.py` dispatches it as a CU command.
3. `src/diaad/core/run_context.py` threads CU paradigms, blinding config, sample identifier column, and excluded speakers.
4. `src/diaad/core/run_wrappers.py` calls `analyze_cu_coding()`.
5. `src/diaad/coding/compl_utts/analysis.py` writes analysis outputs.

## Input Discovery

The command uses exact filename discovery for:

```text
cu_coding.xlsx
```

It searches the input directory and current run output directory.

## Column Detection

The analyzer detects valid SV/REL column pairs from unprefixed, paradigm-suffixed, and coder-prefixed schemas. Administrative coder/comment columns are dropped from analysis outputs.

If `advanced.cu_paradigms` is configured, detected pairs are filtered to those paradigms.

## Output Files

The command writes under:

```text
cu_coding_analysis/
```

Current output filenames are:

```text
cu_coding_by_utterance.xlsx
cu_coding_by_sample_long.xlsx
cu_coding_by_sample.xlsx
cu_analysis_blind_codebook.xlsx
cu_analysis_blinding_diagnostics.xlsx
```

The analysis blind codebook and diagnostics are only written when analysis-stage blinding runs.

## Relevant Sources

- `src/diaad/coding/compl_utts/analysis.py`
- `src/diaad/coding/utils/transcript.py`
- `src/diaad/metadata/blinding.py`
- `src/diaad/metadata/unblinding.py`
- `tests/test_coding/test_compl_utts/test_identifiers.py`
