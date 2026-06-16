# Web App Project Setup and First Run Quickstart

Use the web app when an interactive interface is more convenient than the CLI. Hosted web use can be ideal for learning, teaching, itinerant work, and easily deidentifiable data. Local web use gives the same menu-driven style while keeping files on the user's machine.

## Hosted Web App

If a hosted DIAAD deployment is available, open it in a browser. No Python installation is required.

This route is best for:

- generated examples;
- teaching and demonstrations;
- small interactive runs;
- data that are already deidentified enough for the hosted setting.

## Local Web App

Install the web extra:

```bash
pip install "diaad[web]"
```

Launch the app:

```bash
diaad streamlit
```

Open the printed local URL in a browser.

## First Web Run

In the app:

1. Read the instructions and manual panel.
2. Upload `project.yaml` and `advanced.yaml`, or build configuration in the app.
3. Upload input files or a folder.
4. Select one or more commands in Part 3.
5. Optionally download examples for the selected commands.
6. Run the selected functions.
7. Download the results ZIP before closing the session.

## Choosing Hosted, Local Web, Or CLI

Use hosted web for convenience with deidentified data. Use local web when you want the interface but prefer local control, especially for dialogic samples that may contain explicit or implicit personal information. Use the CLI when the workflow needs automation, scripting, repeated batches, or full local output management.

## Read Next

- Web app operation: `docs/manual/02_operation/03_webapp.md`
- CLI project setup workflow: `docs/manual/06_workflows/01_cli_project_setup_first_run/01_quickstart.md`
- CLI and web execution: `docs/manual/05_functionalities/02_cli_web_execution/01_quickstart.md`
- Example workflow: `docs/manual/06_workflows/03_example_dataset_command_specific_packages/01_quickstart.md`
