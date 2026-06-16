# CLI Project Setup and First Run Implementation Notes

The CLI workflow runs through the main DIAAD entry point and normal run context machinery, except for special top-level commands such as `examples` and `streamlit`.

## Source Anchors

Primary sources:

- `src/diaad/main.py`
- `src/diaad/cli/parser.py`
- `src/diaad/cli/commands.py`
- `src/diaad/cli/dispatch.py`
- `src/diaad/core/config.py`
- `src/diaad/core/config_overrides.py`
- `src/diaad/core/run_context.py`
- `src/diaad/core/provenance.py`

## Command Parsing

Ordinary CLI commands are canonical strings from `MODULE_COMMANDS` in `src/diaad/cli/commands.py`. Multiple commands are parsed from a comma-separated command string:

```bash
diaad "transcripts tabularize, cus files"
```

Unknown command strings are skipped after warning. Examples-specific flags are rejected unless the command is exactly:

```bash
diaad examples
```

## Run Context

For ordinary commands, `src/diaad/main.py` creates a `RunContext`, resolves configuration, writes start artifacts, prepares command prerequisites, builds the dispatch table, and runs each requested command.

For dry-run configuration checks, DIAAD creates the configuration context but does not create a normal output run directory.

## Output Artifacts

Normal CLI runs initialize logging after the timestamped output directory is known. Run finalization writes audit artifacts such as effective configuration, CLI arguments, environment information, directory snapshots, and manifest records.

## Special Commands

`diaad examples` has its own path in `src/diaad/main.py`. It can generate full or command-specific example packages. `diaad examples --render-docs` regenerates packaged Example I/O markdown and returns without creating the normal example package workflow.

`diaad streamlit` launches the local web app through `src/diaad/webapp/launcher.py`.

## Read Next

- Command-line operation: `docs/manual/02_operation/02_command_line.md`
- Configuration implementation notes: `docs/manual/05_functionalities/01_configuration_sources_defaults_overrides/04_implementation_notes.md`
- Run provenance implementation notes: `docs/manual/05_functionalities/03_run_provenance_audit_artifacts/04_implementation_notes.md`
