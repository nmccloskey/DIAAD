# Configuration Sources, Defaults, and Overrides Quickstart

DIAAD configuration starts from packaged defaults, then applies project configuration and command-line overrides. This lets a new user run examples quickly while still supporting project-specific, reproducible analysis settings.

## Start Here

For real projects, keep a split configuration directory in the project root:

```text
config/
  project.yaml
  advanced.yaml
```

Then run DIAAD from that project root:

```bash
diaad transcripts tabularize
```

If a `config/` directory is present, DIAAD uses it automatically. You can also point to a config source explicitly:

```bash
diaad transcripts tabularize --config config
diaad transcripts tabularize --config diaad_config.yaml
```

## Source Order

For CLI runs, DIAAD resolves configuration in this order:

1. Packaged defaults from `src/diaad/config/default_config.yaml`.
2. A user config source, if one is provided or if `./config` exists.
3. CLI overrides from `--input-dir`, `--output-dir`, and `--set KEY=VALUE`.

The final resolved settings are the effective configuration for that run.

## Check Before Running

Use a dry run to inspect the effective configuration:

```bash
diaad transcripts tabularize --dry-run-config --dry-run-config-format yaml
```

Save it when you want an audit copy before processing data:

```bash
diaad transcripts tabularize --dry-run-config --dry-run-config-format yaml --dry-run-config-out resolved_config.yaml
```

## Read Next

- Configuration operation page: `docs/manual/02_operation/04_configuration.md`
- Command-line operation: `docs/manual/02_operation/02_command_line.md`
- Run provenance and audit artifacts: `docs/manual/05_functionalities/03_run_provenance_audit_artifacts/01_quickstart.md`
- Transcript tabularization: `docs/manual/03_features/01_transcript_tabularization.md`
