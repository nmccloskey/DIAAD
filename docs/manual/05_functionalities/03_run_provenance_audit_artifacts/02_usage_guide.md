# Run Provenance and Audit Artifacts Usage Guide

Run provenance artifacts help users reconstruct and audit a DIAAD run. They are especially useful when a project uses configuration overrides, chained commands, reliability sampling, blinding, or repeated batches.

## What To Save

For each important CLI run, save the whole timestamped output directory. At minimum, preserve:

- the `logs/` directory;
- generated outputs needed by downstream commands;
- manually completed coding workbooks, if the run created templates for human coding;
- the project config files used for the run.

Do not rely on memory or command history alone. The effective configuration can differ from project YAML when CLI overrides were used.

## What The Log Files Mean

| Artifact | Use |
|---|---|
| `logs/run_log.log` | Human-readable run log. |
| `logs/run_metadata.json` | Compact summary of run ID, program version, status, commands, runtime, paths, config source, and log artifact names. |
| `logs/effective_config.yaml` | Resolved configuration after defaults, user config, and overrides. |
| `logs/cli_args.json` | Parsed command-line arguments. |
| `logs/config_overrides.json` | Difference introduced by CLI overrides. |
| `logs/directory_snapshot_start.json` | Input and output directory snapshot at run start. |
| `logs/directory_snapshot_end.json` | Input and output directory snapshot at run end. |
| `logs/environment.json` | Captured versions for DIAAD and selected dependencies. |
| `logs/manifest.json` | Run manifest with status, command, artifacts, config source, and produced result files. |

## Produced Results

The manifest records produced result files by comparing start and end snapshots of the run output directory. Log files themselves are excluded from the `results` list so users can focus on substantive outputs.

Use the manifest as a map, then inspect the actual workbooks or files in the output directory.

## Failed Runs

If a CLI run fails after logging starts, DIAAD still attempts to terminate the logger and write final artifacts with failed status. Preserve the run directory, because failed runs often contain useful logs and partial outputs.

If a dry-run config fails, fix the configuration or overrides before running the actual command.

## Web Runs

For web runs, download the output ZIP immediately. The current web path zips the run output directory after selected commands finish. Because web execution does not currently call the same explicit DIAAD provenance artifact writers as normal CLI execution, CLI runs are better when complete audit artifacts are required.

## Recommendations

Use `--dry-run-config` before major processing runs. For final analysis, preserve:

- the original config files;
- the saved dry-run config, if reviewed;
- the run output directory;
- any manually edited files used between runs;
- notes explaining why any CLI overrides were used.

## Read Next

- Run provenance research context: `docs/manual/05_functionalities/03_run_provenance_audit_artifacts/03_research_context.md`
- Configuration usage guide: `docs/manual/05_functionalities/01_configuration_sources_defaults_overrides/02_usage_guide.md`
- Testing: `docs/manual/02_operation/05_testing.md`
