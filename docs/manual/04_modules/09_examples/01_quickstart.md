# Examples Module Quickstart

The Examples module creates synthetic DIAAD projects and supports generated Example I/O documentation. It is useful for learning file layouts, checking command expectations, and maintaining reproducible documentation examples.

## Main Entry Points

| Entry point | Main use |
|---|---|
| `diaad examples` | Generate the full synthetic example dataset. |
| `diaad examples --for-command "<command>"` | Generate a command-specific example package. |
| web app example download | Download full or selected-command example files from the app. |
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

The examples are synthetic. They are not participant data, clinical records, or de-identified real transcripts.

## What To Look For

Example packages show:

- example configuration files;
- minimal input files;
- representative output files;
- an example README;
- `example_manifest.json`;
- runtime-shaped paths that mirror normal DIAAD output conventions.

## Read Next

- [Generated Example I/O](../../03_features/04_generated_example_io.md)
- [Command-line operation](../../02_operation/02_command_line.md)
- [Web app operation](../../02_operation/03_webapp.md)
- [Testing](../../02_operation/05_testing.md)

Later command pages will connect each command to its generated Example I/O view.
