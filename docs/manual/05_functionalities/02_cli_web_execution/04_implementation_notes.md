# CLI and Web Execution Implementation Notes

CLI and web execution share the command registry and dispatch table for ordinary DIAAD commands. Special commands such as `examples` and `streamlit` are handled before ordinary dispatch.

## Source Anchors

Primary sources:

- `src/diaad/main.py`
- `src/diaad/cli/commands.py`
- `src/diaad/cli/parser.py`
- `src/diaad/cli/dispatch.py`
- `src/diaad/core/run_context.py`
- `src/diaad/core/run_wrappers.py`
- `src/diaad/webapp/streamlit_app.py`
- `src/diaad/webapp/config_builder.py`
- `src/diaad/webapp/launcher.py`

Relevant tests:

- `tests/test_main.py`
- `tests/test_cli/test_parser.py`
- `tests/test_webapp_command_menu.py`
- `tests/test_webapp_launcher.py`

## CLI Path

The normal CLI path is:

1. `src/diaad/cli/parser.py` parses positional commands and common options.
2. `src/diaad/main.py` builds config overrides and initializes `RunContext`.
3. `parse_cli_commands()` normalizes and validates ordinary commands against `MODULE_COMMANDS`.
4. Dry-run config exits before logger initialization and dispatch.
5. Normal runs initialize logging, write start artifacts, prepare prerequisites, build dispatch, execute requested commands, and finalize run artifacts through logger termination hooks.

Special cases:

- `diaad streamlit` launches the web app and does not create `RunContext`.
- `diaad examples` has its own generation path and creates a normal `RunContext` unless `--render-docs` is used.
- `diaad examples --render-docs` bypasses normal run context and refreshes packaged generated documentation.

## Dispatch And Prerequisites

`build_dispatch(ctx)` maps canonical command strings to zero-argument callables that close over the current `RunContext`.

`prepare_dispatch_prerequisites(ctx, commands)` loads shared prerequisites before ordinary dispatch:

- commands that require CHAT files call `ctx.load_chats()`;
- commands that require transcript tables call `ctx.ensure_transcript_tables()` unless `transcripts tabularize` is part of the same run.

If transcript tables are required but absent, `ctx.ensure_transcript_tables()` raises an error unless `auto_tabularize` is true.

## Web Path

The web app path is:

1. Streamlit renders a configuration upload/builder UI.
2. Uploaded or built config is written into a temporary split config directory.
3. Uploaded inputs are saved under a temporary input directory, preserving safe nested paths.
4. `RunContext` is initialized with that temporary project root.
5. Selected commands are prepared and dispatched.
6. The timestamped run output directory is zipped and offered for download.

The web command menu is built from `MODULE_COMMANDS`, filtered through dispatch support, and excludes `examples` from ordinary command selection. Examples are generated through a dedicated web action.

## Provenance Difference

Normal CLI runs call `write_start_artifacts()` and finalize through `finalize_run_artifacts()`. The current web run path initializes and terminates logging, but it does not explicitly call the same DIAAD provenance artifact writers.

TODO: Confirm whether future web runs should write the same `effective_config.yaml`, `manifest.json`, and directory snapshots as CLI runs, or whether the manual should continue documenting web outputs as a lighter ZIP-based execution record.

## Command Registry Concern

The current source registry includes `templates combine`, and dispatch supports it. The authored manual now includes a command page for it. Before final publication, keep the registry, dispatch table, web command menu, and command documentation synchronized so user-facing command lists do not drift.

## Read Next

- Run provenance implementation notes: `docs/manual/05_functionalities/03_run_provenance_audit_artifacts/04_implementation_notes.md`
- Configuration implementation notes: `docs/manual/05_functionalities/01_configuration_sources_defaults_overrides/04_implementation_notes.md`
- Command-line operation: `docs/manual/02_operation/02_command_line.md`
- Web app operation: `docs/manual/02_operation/03_webapp.md`
