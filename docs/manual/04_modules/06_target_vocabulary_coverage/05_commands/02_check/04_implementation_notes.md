# `vocab check` Implementation Notes

`vocab check` dispatches to `check_target_vocab_resources()`.

## Dispatch Path

The main path is:

1. `src/diaad/cli/commands.py` registers `vocab check`.
2. `src/diaad/cli/dispatch.py` dispatches it as a target-vocabulary command.
3. `src/diaad/core/run_context.py` threads the configured resource path and output path.
4. `src/diaad/core/run_wrappers.py` calls `check_target_vocab_resources()`.
5. `src/diaad/coding/target_vocab/files.py` writes the check report.

## Resource Loading

Resource loading starts with bundled JSON files under:

```text
src/diaad/coding/target_vocab/resources/
```

If `advanced.target_vocabulary_resource_path` is configured, DIAAD loads custom resources from that file or directory and merges them with the built-ins. Custom resources with resource IDs matching bundled resources override those bundled definitions.

## Validation Boundary

Validation is implemented in `src/diaad/coding/target_vocab/resources.py`. It checks resource shape, base-form and variant consistency, and norm specification shape. It does not load online norms and does not validate the research interpretation of a resource.

## Output Path

The report is written to:

```text
target_vocab/
  target_vocab_resource_check.txt
```

under the current run output directory.

## Relevant Sources

- `src/diaad/coding/target_vocab/files.py`
- `src/diaad/coding/target_vocab/resources.py`
- `tests/test_coding/test_target_vocab/test_files.py`
- `tests/test_coding/test_target_vocab/test_resources.py`
