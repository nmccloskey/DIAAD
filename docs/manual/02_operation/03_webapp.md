# Web App Operation

The DIAAD web app provides an interactive way to build or upload configuration, upload input files, choose commands, run DIAAD, and download a zipped output folder. It is designed for accessibility and exploration; the command-line interface remains the better option for sensitive data, large projects, repeated pipelines, and highly customized local file handling.

## Access Options

If a hosted DIAAD deployment is available for your project, open it in a browser and follow the upload workflow.

To run the web app locally, install the web extra:

```bash
pip install "diaad[web]"
```

Then launch:

```bash
streamlit_diaad
```

The launcher starts Streamlit and prints a local URL. Open that URL in a browser to use the app.

## Privacy Note

The web app runs each analysis in a temporary workspace and returns outputs as a ZIP file. That behavior supports clean web sessions, but it is not a substitute for data governance. Use appropriately deidentified data in hosted deployments, and prefer local CLI processing for identifiable, sensitive, or hard-to-deidentify discourse data.

## Configuration

The web app supports two configuration paths:

- upload both `project.yaml` and `advanced.yaml`; or
- build a configuration in the app and choose the built configuration for the run.

The web app expects the split configuration form. Uploading only one of the two files is incomplete.

During a web run, DIAAD uses the app's temporary input and output folders. This means uploaded configuration values for `project.input_dir` and `project.output_dir` are not used as local filesystem paths in the browser session. Other project and advanced settings still control processing behavior. See Configuration (`docs/manual/02_operation/04_configuration.md`) for the shared configuration model and current settings.

## Uploading Inputs

Upload the files or folder needed for the selected commands. The web app accepts common DIAAD input types, including:

- `.cha`
- `.xlsx`
- `.csv`
- `.json`

When uploading a folder, preserve the intended directory structure. The current web workflow keeps nested uploaded paths so command-specific inputs can remain organized.

## Selecting Commands

The command menu is organized by DIAAD module. Select the command or commands you want to run, then start the run from the app.

The `examples` command is handled separately from the normal command menu. Use the app's example download control to obtain generated example files.

If no commands are selected, the examples control downloads the full example dataset. If commands are selected, it downloads example files relevant to the selected functions.

## Running and Downloading

A web run requires:

- a complete configuration;
- at least one uploaded input file;
- at least one selected DIAAD command.

After the run finishes, download the result ZIP. Web outputs are packaged with a timestamped name such as:

```text
diaad_web_output_YYMMDD_HHMM.zip
```

Download outputs before closing or refreshing the browser session.

## Web App Versus CLI

| Area | Web app | CLI |
|---|---|---|
| Setup | Browser workflow, or local `diaad[web]` install | Local Python environment |
| Configuration | Upload split config or build in app | Packaged defaults, split config, nested YAML, and command-line overrides |
| Input/output paths | Temporary web workspace and downloadable ZIP | User-controlled local directories |
| Command selection | Interactive module menu | Typed commands and chained command strings |
| Examples | Dedicated download control | `diaad examples` and examples-specific flags |
| Best fit | Learning, deidentified examples, smaller interactive runs | Sensitive data, repeated workflows, scripted runs, large projects |

## Practical Recommendations

Use generated examples in the web app before uploading study data. Keep local copies of the exact `project.yaml` and `advanced.yaml` used for a run. Download the output ZIP immediately after a successful run. For workflows that depend on manually completed coding files, use the web app for one stage at a time so you can inspect intermediate outputs before uploading them for later stages.

## Common Problems

If the app reports incomplete configuration, upload both `project.yaml` and `advanced.yaml`, or complete the in-app configuration builder and select the built configuration.

If the app cannot run, confirm that at least one input file and at least one command are selected.

If a later-stage command cannot find expected files, check that the uploaded folder preserves the output structure from the earlier stage and that configured filenames match the files being uploaded.

If a workflow needs NLP support and the hosted app does not provide the required model, run the workflow locally with `diaad[nlp]` and the configured spaCy language model installed.
