# Run Provenance and Audit Artifacts Quickstart

DIAAD writes run artifacts so users can inspect what happened during a CLI run: which commands ran, which configuration was effective, which overrides were applied, what files existed at the beginning and end, and which result files were produced.

## Normal CLI Runs

A normal CLI run creates a timestamped output directory:

```text
diaad_data/output/diaad_YYMMDD_HHMM/
```

Inside that run directory, inspect:

```text
logs/
  run_log.log
  run_metadata.json
  effective_config.yaml
  cli_args.json
  config_overrides.json
  directory_snapshot_start.json
  directory_snapshot_end.json
  environment.json
  manifest.json
```

## Dry-Run Config

Before processing data, inspect the effective configuration:

```bash
diaad transcripts tabularize --dry-run-config --dry-run-config-format yaml
```

Save the dry-run payload when reviewing settings before a run:

```bash
diaad transcripts tabularize --dry-run-config --dry-run-config-format yaml --dry-run-config-out resolved_config.yaml
```

Dry-run config output does not run the requested command.

## Web App

The web app returns a ZIP of the run output. Treat the ZIP as the practical record of the web session, but do not assume it currently contains the same complete DIAAD provenance artifact set as a normal CLI run.

## Read Next

- Run provenance usage guide: `docs/manual/05_functionalities/03_run_provenance_audit_artifacts/02_usage_guide.md`
- Configuration quickstart: `docs/manual/05_functionalities/01_configuration_sources_defaults_overrides/01_quickstart.md`
- Command-line operation: `docs/manual/02_operation/02_command_line.md`
- Web app operation: `docs/manual/02_operation/03_webapp.md`
