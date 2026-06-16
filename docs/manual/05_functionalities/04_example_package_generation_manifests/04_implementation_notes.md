# Example Package Generation and Manifests Implementation Notes

Example package generation is implemented in `src/diaad/examples/generate.py` and invoked from `src/diaad/main.py` and the web app.

## Source Anchors

Primary sources:

- `src/diaad/examples/generate.py`
- `src/diaad/examples/assets/spec/`
- `src/diaad/main.py`
- `src/diaad/webapp/streamlit_app.py`

Relevant tests:

- `tests/test_examples/test_examples.py`
- `tests/test_webapp_command_menu.py`
- `tests/test_main.py`

## CLI Path

`src/diaad/main.py` detects `diaad examples` before ordinary dispatch.

For user-facing package generation, it:

1. creates a `RunContext`;
2. initializes logging;
3. writes normal start artifacts;
4. calls `generate_example_files()`;
5. finalizes run artifacts.

When `--for-command` is omitted, the destination is:

```text
<run_output_dir>/example_files_full_dataset/
```

When `--for-command` is present, the destination is the run output directory, and the generator creates:

```text
example_files_<slug>/
```

## Web Path

The web app calls `_run_examples_web()` from `src/diaad/webapp/streamlit_app.py`.

With no selected commands, it calls:

```text
generate_example_files(examples_root / example_package_name(), force=True)
```

With selected commands, it calls:

```text
generate_example_files(examples_root, force=True, commands=commands)
```

The generated examples root is zipped and returned through a download button. Web examples are generated through this dedicated action rather than through the ordinary command dispatch menu.

## Package Naming

`generate.py` defines:

```text
EXAMPLE_PACKAGE_PREFIX = example_files_
FULL_DATASET_SLUG = full_dataset
MANIFEST_FILENAME = example_manifest.json
RUN_DIR = diaad_YYMMDD_HHMM
```

Canonical commands become stable IDs and slugs:

```text
cus files -> cus.files -> cus_files
```

Full examples use:

```text
example_files_full_dataset
```

Command-specific examples use:

```text
example_files_<command_slug_or_command_set_slug>
```

## Example Specs

Synthetic data and example configuration are read from:

```text
src/diaad/examples/assets/spec/
```

Important spec files include:

- `dataset.yaml`
- `configs/project.yaml`
- `configs/advanced.yaml`
- transcript specs;
- template subset specs;
- target vocabulary resource specs;
- digital turn session specs.

Command-specific package configs are derived from these specs, with `input_dir` rewritten to `example_input` and `output_dir` rewritten to `example_output`.

## Command Plans And Capabilities

`EXAMPLE_COMMAND_PLANS` maps supported canonical commands to `ExampleCommandPlan` records. Each plan declares:

- the canonical command;
- required input capabilities;
- the output builder;
- optional output capabilities that later requested commands can reuse.

`_required_capabilities_for_commands()` deduplicates shared inputs and accounts for capabilities produced by earlier commands in the same requested package.

Unsupported commands raise a `ValueError` listing available example commands. `templates subset` raises a separate error if requested with other commands.

## Manifests

`_write_full_example_manifest()` writes full-dataset manifest metadata, including:

- `package_kind`;
- `workflow_id`;
- `command_id`;
- `covered_commands`;
- `command_ids`;
- directory names;
- config, input, and expected output artifacts;
- notes about the `expected_outputs/` atlas.

`_write_command_example_manifest()` writes command-specific manifest metadata, including:

- `package_kind`;
- command metadata;
- command IDs;
- required capabilities;
- suggested invocation;
- package directories;
- config, input, output, and log artifacts.

Artifact records include preview kinds such as workbook, text, JSON, YAML, file, or directory.

## Runtime And Atlas Paths

The generator keeps several path contexts distinct:

- full-dataset atlas path, such as `expected_outputs/cus_module/cus_files/cu_coding.xlsx`;
- command-package preview path, such as `example_output/cu_coding/cu_coding.xlsx`;
- normal runtime display path, such as `diaad_data/output/diaad_YYMMDD_HHMM/cu_coding/cu_coding.xlsx`.

This prevents the full-dataset atlas layout from being mistaken for ordinary DIAAD output structure.

## Maintenance Notes

When example package behavior changes, update the specs, generator helpers, manifests, and tests together. Avoid hand-editing generated packages or generated Example I/O markdown.

The current example command plan registry does not include `templates combine`, although `templates combine` exists in the CLI registry and dispatch. Decide whether to add an example plan for it, omit it intentionally, or change its user-facing status.

## Read Next

- Generated Example I/O implementation notes: `docs/manual/05_functionalities/05_generated_example_io_manual_composition/04_implementation_notes.md`
- Examples command implementation notes: `docs/manual/04_modules/09_examples/05_commands/01_examples/04_implementation_notes.md`
- Testing: `docs/manual/02_operation/05_testing.md`
