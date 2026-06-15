# `vocab file` Implementation Notes

`vocab file` dispatches to `make_target_vocab_file()`.

## Dispatch Path

The main path is:

1. `src/diaad/cli/commands.py` registers `vocab file`.
2. `src/diaad/cli/dispatch.py` dispatches it as a target-vocabulary command.
3. `src/diaad/core/run_context.py` threads input and output paths.
4. `src/diaad/core/run_wrappers.py` calls `make_target_vocab_file()`.
5. `src/diaad/coding/target_vocab/files.py` writes the JSON template.

## Template Construction

The template is built from the shape of a bundled resource. This keeps the blank custom template aligned with the currently supported resource fields, including the norm specification shape used by bundled resources.

## Output Path

The command writes:

```text
target_vocab/
  target_vocabulary_resource_template.json
```

under the current run output directory.

## Relevant Sources

- `src/diaad/coding/target_vocab/files.py`
- `src/diaad/coding/target_vocab/resources.py`
- `tests/test_coding/test_target_vocab/test_files.py`
