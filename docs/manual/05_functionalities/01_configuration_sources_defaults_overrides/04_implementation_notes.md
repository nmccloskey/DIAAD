# Configuration Sources, Defaults, and Overrides Implementation Notes

Configuration is normalized through `ConfigManager` and then carried through a `RunContext`.

## Source Anchors

Primary sources:

- `src/diaad/core/config.py`
- `src/diaad/core/config_overrides.py`
- `src/diaad/core/run_context.py`
- `src/diaad/config/default_config.yaml`
- `src/diaad/cli/parser.py`
- `src/diaad/webapp/config_builder.py`

Relevant tests:

- `tests/test_core/test_config.py`
- `tests/test_core/test_config_overrides.py`
- `tests/test_core/test_run_context.py`
- `tests/test_cli/test_parser.py`

## Loading And Merge Order

`RunContext._resolve_config_source()` decides which user config source to pass to `ConfigManager`:

- an explicit `--config` path is resolved relative to the project root if needed;
- if `--config` is omitted and `./config` exists, that directory is used;
- otherwise `None` is passed, and packaged defaults are used.

`ConfigManager` then:

1. loads `src/diaad/config/default_config.yaml`;
2. loads a split directory, nested YAML file, or no user config;
3. merges defaults with user settings;
4. applies CLI overrides;
5. parses and validates normalized `ProjectConfig` and `AdvancedConfig` dataclasses;
6. records `config_source` metadata and `override_diff`.

The configuration sections are:

```text
project
advanced
```

## Supported Config Shapes

The code supports:

- split config directories containing `project.yaml` and `advanced.yaml`;
- a nested YAML file with top-level `project:` and `advanced:` sections;
- a directory containing a single nested `config.yaml`;
- missing sections, which are filled from packaged defaults.

An explicit missing config path raises `FileNotFoundError`. A directory that mixes split files and nested config in an ambiguous way raises `ValueError`.

## Overrides

CLI overrides are parsed in `src/diaad/core/config_overrides.py`.

Direct flags are converted to canonical keys:

```text
--input-dir  -> project.input_dir
--output-dir -> project.output_dir
```

Repeated `--set KEY=VALUE` values are parsed by PSAIR helper logic, then DIAAD maps supported keys to either `project` or `advanced`. Unknown keys raise an error. Explicit sectioned keys such as `advanced.transcript_table_filename` are clearest.

The direct input/output flags take precedence over conflicting `--set input_dir=...` and `--set output_dir=...` values because `build_cli_config_overrides()` applies them after parsed `--set` values.

## Validation

Important validation and normalization behavior includes:

- `reliability_fraction` must be greater than `0` and less than or equal to `1`;
- `num_bins` must be at least `1`;
- `sample_id_column`, `utterance_id_column`, and `spacy_model_name` must be non-empty strings;
- `id_columns` defaults to `sample_id` and `utterance_id` and removes duplicates while preserving order;
- boolean-like strings are normalized where supported;
- list fields must be YAML lists unless an override parser supplies an already parsed value.

## Web Config Builder

The web app writes split `project.yaml` and `advanced.yaml` files into a temporary `config/` directory before constructing `RunContext`. `_web_project_config()` rewrites `input_dir` and `output_dir` to web-session folders:

```text
input
output
```

The config builder's UI defaults are declared in `src/diaad/webapp/config_builder.py`. Some are starter values for a usable interface and do not exactly match `default_config.yaml`.

TODO: Review whether the web builder should explicitly label these as presets in the app and manual.

## Boundary With PSAIR

DIAAD uses PSAIR helpers for sectioned config loading, default merging, dry-run payload formatting, and override parsing. DIAAD manual pages should explain the user-visible behavior. PSAIR-specific helper contracts belong in PSAIR documentation unless a DIAAD user needs the detail for troubleshooting.

## Read Next

- Configuration operation page: `docs/manual/02_operation/04_configuration.md`
- Run provenance implementation notes: `docs/manual/05_functionalities/03_run_provenance_audit_artifacts/04_implementation_notes.md`
- Testing: `docs/manual/02_operation/05_testing.md`
