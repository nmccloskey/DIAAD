# Run Provenance and Audit Artifacts Implementation Notes

Run provenance is implemented in `src/diaad/core/provenance.py` and called from `src/diaad/main.py` for normal CLI runs and user-facing `diaad examples` generation.

## Source Anchors

Primary sources:

- `src/diaad/core/provenance.py`
- `src/diaad/main.py`
- `src/diaad/core/run_context.py`
- `src/diaad/core/config.py`

Relevant tests:

- `tests/test_core/test_provenance.py`
- `tests/test_main.py`
- `tests/test_core/test_run_context.py`

## Dry-Run Payload

`build_dry_run_payload()` records:

- program name and version;
- requested commands;
- run paths;
- serialized CLI arguments;
- config override diff;
- effective configuration;
- selected dependency environment information;
- config source metadata.

`emit_dry_run_config()` prints the payload as JSON or YAML and optionally writes it to `--dry-run-config-out`.

In `main.py`, dry-run config initializes `RunContext` with `create_output_dir=False`, emits the payload, and exits before logger initialization and command dispatch.

## Start Artifacts

`write_start_artifacts()` creates `logs/` and writes:

```text
logs/directory_snapshot_start.json
logs/effective_config.yaml
logs/cli_args.json
logs/config_overrides.json
logs/environment.json
```

The start snapshot captures input directory contents and the current run output directory contents, rooted relative to the project or run output directory where possible.

## Final Artifacts

`finalize_run_artifacts()` writes:

```text
logs/directory_snapshot_end.json
logs/manifest.json
logs/run_metadata.json
```

The manifest combines fixed log artifact paths with a `results` list derived from files present at end but not start. Result paths that begin with `logs/` are excluded.

`run_metadata.json` includes:

- run ID;
- DIAAD version;
- status;
- commands;
- start and end timestamps;
- runtime seconds;
- run paths;
- config source metadata;
- log artifact names.

## Logging Hook

Normal CLI runs call:

```text
add_finalization_hook(lambda context: finalize_run_artifacts(ctx, context))
```

Then `terminate_logger()` triggers finalization in the `finally` block. This means final artifacts are attempted for completed and failed runs after logger initialization.

## Special Cases

`diaad examples` creates a normal `RunContext`, initializes logging, writes start artifacts, and finalizes run artifacts for user-facing generated example packages.

`diaad examples --render-docs` bypasses normal run context and calls `render_example_docs()` directly. It does not create normal run provenance artifacts.

`diaad streamlit` launches the web app without creating `RunContext`.

The Streamlit web run path currently creates `RunContext`, initializes logging, dispatches selected commands, terminates logging, and zips the run output. It does not explicitly call `write_start_artifacts()` or register `finalize_run_artifacts()`.

## Boundary With PSAIR

DIAAD delegates low-level serialization, environment capture, config dry-run formatting, and manifest writing to PSAIR helpers. DIAAD docs should describe the files users see and how to interpret them. PSAIR docs can cover reusable helper internals.

## Read Next

- CLI and web execution implementation notes: `docs/manual/05_functionalities/02_cli_web_execution/04_implementation_notes.md`
- Configuration implementation notes: `docs/manual/05_functionalities/01_configuration_sources_defaults_overrides/04_implementation_notes.md`
- Testing: `docs/manual/02_operation/05_testing.md`
