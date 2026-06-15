# `examples` Implementation Notes

`diaad examples` is handled before normal command dispatch.

## CLI Path

The main user-facing path is:

1. `src/diaad/cli/parser.py` defines `--for-command`, `--force`, and `--render-docs`.
2. `src/diaad/main.py` detects when the positional command is `examples`.
3. For user-facing file generation, `main.py` creates a normal `RunContext`, initializes logging, writes run artifacts, and calls `generate_example_files()`.
4. `src/diaad/examples/generate.py` writes either the full package or a command-specific package.

Examples-specific flags are valid only with `diaad examples`. The normal dispatch table does not expose `examples` as a webapp command checkbox or as a regular module/action dispatch callable.

## Full Package Generation

When no command-specific examples are requested, the CLI calls `generate_example_files()` with a destination ending in:

```text
example_files_full_dataset
```

The full package uses packaged specs and writes:

```text
config/
input/
expected_outputs/
example_manifest.json
```

The manifest declares `package_kind: full_dataset`, `workflow_id: full_example_dataset`, and `command_id: examples`.

## Command-Specific Generation

When `--for-command` is supplied, each flag value is passed as one canonical command. The generator normalizes commands, validates them against the CLI command registry and `EXAMPLE_COMMAND_PLANS`, and writes a package named:

```text
example_files_<slug>
```

The command-specific package uses:

```text
example_config/
example_input/
example_output/
example_logs/
example_manifest.json
```

The manifest declares `package_kind: command_specific`, command IDs, required capabilities, output artifacts, runtime display paths, and suggested invocation.

The generator can reuse outputs from earlier requested commands as capabilities for later requested commands in the same package.

## Web App Path

The web app has a dedicated examples download action in Part 3: Commands. `_run_examples_web()` generates packages in a temporary examples root and returns a ZIP.

With no selected commands, the ZIP contains `example_files_full_dataset/`. With selected commands, the ZIP contains the command-specific `example_files_<slug>/` package. The ZIP filename uses:

```text
diaad_example_files_<slug>_YYMMDD_HHMM.zip
```

## Rendered Example I/O

`--render-docs` is maintainer-facing. It bypasses the normal `RunContext` path and calls `render_example_docs()`, which writes generated markdown under:

```text
src/diaad/examples/assets/rendered_docs/example_io/
```

Generated command pages receive composable front matter with fields such as `object_type`, `object_id`, `command_id`, `canonical_command`, `module_id`, `view: example_io`, `view_order`, `slot`, and `source_manual`.

The full example dataset overview is modeled primarily as a workflow object with secondary command identity for `examples`.

## Relevant Sources

- `src/diaad/main.py`
- `src/diaad/cli/parser.py`
- `src/diaad/examples/generate.py`
- `src/diaad/examples/render_docs.py`
- `src/diaad/webapp/streamlit_app.py`
- `tests/test_examples/test_examples.py`
- `tests/test_webapp_command_menu.py`
