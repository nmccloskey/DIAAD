# Web App Project Setup and First Run Implementation Notes

The web app is implemented with Streamlit and uses DIAAD's ordinary command registry for dispatchable commands.

## Source Anchors

Primary sources:

- `src/diaad/webapp/streamlit_app.py`
- `src/diaad/webapp/config_builder.py`
- `src/diaad/webapp/launcher.py`
- `src/diaad/main.py`
- `src/diaad/cli/commands.py`
- `tests/test_webapp_command_menu.py`

## Launch Path

`diaad streamlit` is handled as a special top-level command in `src/diaad/main.py`. It calls `launch_streamlit()` in `src/diaad/webapp/launcher.py`.

The launcher checks whether Streamlit is installed. If not, it prints an install message for:

```bash
pip install "diaad[web]"
```

When Streamlit is available, the launcher runs the installed web app script with the active Python environment.

## Web App Structure

The app is organized into parts:

- instructions and manual display;
- configuration upload or builder;
- input file or folder upload;
- command selection and examples download;
- output ZIP download after a run.

The command menu is built from `MODULE_COMMANDS`, excluding `examples` as an ordinary dispatch command. Tests assert that the web command menu matches the CLI registry except for the examples special case.

## Temporary Workspaces

Web runs use temporary input and output directories. Uploaded folder structure is preserved where safe. The result is returned as a ZIP file.

The examples download path also uses a temporary directory, then zips either the full example dataset or selected command-specific packages.

## Boundary

The web app is an execution surface. Command behavior remains implemented in the module and command source files. Workflow pages should therefore link to command and functionality pages for detailed file contracts.

## Read Next

- Web app operation: `docs/manual/02_operation/03_webapp.md`
- CLI and web execution implementation notes: `docs/manual/05_functionalities/02_cli_web_execution/04_implementation_notes.md`
- Examples workflow implementation notes: `docs/manual/06_workflows/03_example_dataset_command_specific_packages/04_implementation_notes.md`
