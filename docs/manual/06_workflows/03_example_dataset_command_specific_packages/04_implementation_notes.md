# Example Dataset and Command-Specific Packages Implementation Notes

The examples workflow is implemented as a special DIAAD command path plus a web-app download path.

## Source Anchors

Primary sources:

- `src/diaad/main.py`
- `src/diaad/cli/parser.py`
- `src/diaad/examples/generate.py`
- `src/diaad/examples/render_docs.py`
- `src/diaad/examples/assets/spec/`
- `src/diaad/examples/assets/rendered_docs/example_io/`
- `src/diaad/webapp/streamlit_app.py`
- `tests/test_examples/test_examples.py`
- `tests/test_webapp_command_menu.py`

## CLI Path

`diaad examples` is detected before ordinary dispatch in `src/diaad/main.py`.

Without `--for-command`, DIAAD writes the full dataset package under the current run output directory:

```text
example_files_full_dataset/
```

With one or more `--for-command` values, DIAAD passes those canonical commands to `generate_example_files()` and writes a command-specific package.

Examples-specific flags are valid only with `diaad examples`:

```text
--for-command
--force
--render-docs
```

## Web Path

The web app excludes `examples` from the ordinary command menu. Instead, Part 3 provides a dedicated examples download action.

If no commands are selected, `_run_examples_web()` generates the full dataset. If commands are selected, it generates command-specific examples for those commands. The result is zipped for browser download.

## Package Shapes

Full packages are atlas-shaped and include:

```text
config/
input/
expected_outputs/
example_manifest.json
```

Command-specific packages are toy-project-shaped and include:

```text
example_config/
example_input/
example_output/
example_logs/
example_manifest.json
```

Tests assert package structure, manifest metadata, supported command behavior, and several command-specific output paths.

## Rendered Example I/O

`diaad examples --render-docs` calls `render_example_docs()` and regenerates packaged Markdown under:

```text
src/diaad/examples/assets/rendered_docs/example_io/
```

This is a documentation maintenance path. Ordinary users generally want `diaad examples` or `diaad examples --for-command "<command>"`.

## Read Next

- Examples command implementation notes: `docs/manual/04_modules/09_examples/05_commands/01_examples/04_implementation_notes.md`
- Example package generation implementation notes: `docs/manual/05_functionalities/04_example_package_generation_manifests/04_implementation_notes.md`
- Generated Example I/O implementation notes: `docs/manual/05_functionalities/05_generated_example_io_manual_composition/04_implementation_notes.md`
