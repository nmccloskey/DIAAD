# Examples Module Quickstart

The Examples module creates synthetic DIAAD projects and supports generated Example I/O documentation. It is useful for learning file layouts, checking command expectations, trying toy data before study data, and maintaining reproducible documentation examples.

## Main Entry Points

| Entry point | Main use |
|---|---|
| `diaad examples` | Generate the full synthetic example dataset. |
| `diaad examples --for-command "<command>"` | Generate a command-specific example package. |
| web app Part 3 example download | Download full or selected-command example files from the app. |
| `diaad examples --render-docs` | Regenerate packaged Example I/O documentation for maintainers. |

## Typical User Start

Generate the full example dataset:

```bash
diaad examples --config config
```

Generate examples for one command:

```bash
diaad examples --for-command "transcripts tabularize" --config config
```

Generate examples for multiple specific commands by repeating `--for-command`:

```bash
diaad examples --for-command "cus analyze" --for-command "cus evaluate" --config config
```

The examples are synthetic. They are not participant data, clinical records, or de-identified real transcripts.

## Full And Specific Example Sets

The full example dataset is comprehensive. It pools synthetic inputs and organizes expected outputs by module and command, making it useful as an atlas of DIAAD behavior. Because of that atlas structure, it maps less directly onto a single expected user workflow.

Command-specific packages are smaller toy projects. They use `example_config/`, `example_input/`, `example_output/`, and `example_logs/` so users can inspect a concrete package that more closely resembles one command's expected use case.

## What To Look For

Example packages show:

- example configuration files;
- minimal input files;
- representative output files;
- an example README;
- `example_manifest.json`;
- runtime-shaped paths that mirror normal DIAAD output conventions.

The package configs are generated from DIAAD's packaged example specs. They are close to ordinary DIAAD configuration, but they intentionally set demonstration-friendly values such as synthetic metadata fields, `num_bins`, `num_coders`, `auto_blind`, and the custom target-vocabulary resource path. See Configuration (`docs/manual/02_operation/04_configuration.md`) for DIAAD's current defaults.

## Read Next

- Generated Example I/O: `docs/manual/03_features/04_generated_example_io.md`
- Command-line operation: `docs/manual/02_operation/02_command_line.md`
- Web app operation: `docs/manual/02_operation/03_webapp.md`
- Configuration: `docs/manual/02_operation/04_configuration.md`
- Testing: `docs/manual/02_operation/05_testing.md`

The command page gives exact CLI and webapp usage for generating examples.
