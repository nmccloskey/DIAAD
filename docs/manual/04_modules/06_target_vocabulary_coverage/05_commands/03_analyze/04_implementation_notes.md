# `vocab analyze` Implementation Notes

`vocab analyze` dispatches to `run_target_vocab()`.

## Dispatch Path

The main path is:

1. `src/diaad/cli/commands.py` registers `vocab analyze`.
2. `src/diaad/cli/dispatch.py` currently marks it as requiring transcript tables before dispatch.
3. `src/diaad/core/run_context.py` threads metadata fields, excluded speakers, stimulus column, resource path, sample identifier, and transcript table filename.
4. `src/diaad/core/run_wrappers.py` calls `run_target_vocab()`.
5. `src/diaad/coding/target_vocab/analysis.py` writes the analysis workbook.

## Input Discovery

After the CLI prerequisite check, the analysis implementation searches the current run output directory and input directory for one preferred unblinded utterance file:

```text
unblind_utterance_data*.xlsx
```

If that file is unavailable, it uses the configured transcript table filename.

## Resource Loading

Resources are loaded with `load_target_vocabulary_resources()`. Built-ins are loaded first. Custom resources are merged in when configured, and custom resource IDs override built-in resource IDs when they overlap.

Only input rows whose stimulus or narrative value is in the active resource ID set are analyzed.

## Matching And Output

For each sample, utterances are concatenated and normalized with `reformat()`. Tokens are matched through the resource's reverse variant lookup. The summary sheet is written with ordered target-vocabulary metrics, and the details sheet is written with one row per base form.

The output filename is timestamped:

```text
target_vocab_data_YYMMDD_HHMM.xlsx
```

## Relevant Sources

- `src/diaad/coding/target_vocab/analysis.py`
- `src/diaad/coding/target_vocab/utils.py`
- `src/diaad/coding/target_vocab/resources.py`
- `tests/test_coding/test_target_vocab/test_analysis.py`
- `tests/test_coding/test_target_vocab/test_utils.py`
- `tests/test_coding/test_target_vocab/test_identifiers.py`
