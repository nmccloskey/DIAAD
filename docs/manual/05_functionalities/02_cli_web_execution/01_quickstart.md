# CLI and Web Execution Quickstart

DIAAD can run from the command line or through the Streamlit web app. Both routes use the same core command registry and dispatch layer for ordinary commands, but they differ in configuration, file handling, examples, and audit artifacts.

## CLI

Most commands use:

```bash
diaad <module> <action>
```

Examples:

```bash
diaad transcripts tabularize
diaad cus analyze
diaad powers rates
```

Multiple ordinary commands can be run in one invocation:

```bash
diaad "transcripts tabularize, cus files, words files"
```

Use the CLI for sensitive data, repeated workflows, large projects, scripted runs, and local audit trails.

## Web App

Launch the local web app with:

```bash
diaad streamlit
```

In the web app:

1. Upload or build configuration.
2. Upload input files.
3. Select commands in Part 3.
4. Run selected functions.
5. Download the output ZIP.

The web app is best for learning, smaller interactive runs, and deidentified examples.

## Examples

`diaad examples` is not a normal dispatch command in the web command menu. It has its own CLI path and a dedicated examples download action in the web app.

## Read Next

- Command-line operation: `docs/manual/02_operation/02_command_line.md`
- Web app operation: `docs/manual/02_operation/03_webapp.md`
- Configuration: `docs/manual/05_functionalities/01_configuration_sources_defaults_overrides/01_quickstart.md`
- Run provenance and audit artifacts: `docs/manual/05_functionalities/03_run_provenance_audit_artifacts/01_quickstart.md`
