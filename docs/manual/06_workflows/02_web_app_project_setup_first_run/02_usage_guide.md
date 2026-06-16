# Web App Project Setup and First Run Usage Guide

The web app is a workflow surface for users who want menus, uploads, downloads, and an in-browser configuration builder. It can be hosted remotely or launched locally with `diaad streamlit`.

## Pick The Web Mode

| Mode | Best fit |
|---|---|
| Hosted web app | No-install use, teaching, demos, generated examples, and small deidentified projects. |
| Local web app | Menu-driven local use, sensitive projects where users prefer local control, and workflows that do not need scripting. |
| CLI | Scripted runs, automated pipelines, large datasets, detailed local file management, and repeated batches. |

The hosted app can be quite safe in the sense that the Streamlit session is temporary and users download the outputs. Still, hosted processing means the uploaded data leave the user's local machine during the session. For hard-to-deidentify discourse, local web or CLI use may feel and function more appropriate.

## Configure The Run

The web app accepts split configuration files:

```text
project.yaml
advanced.yaml
```

It can also build configuration interactively. Uploaded path values for `input_dir` and `output_dir` are replaced by temporary web-session folders, but the other settings still matter.

Keep a local copy of the configuration used for a real run. If the app built the configuration, download or preserve it with the project notes.

## Upload Inputs

Upload files or a folder in the structure expected by the selected command. The app preserves nested uploaded paths under its temporary input directory when possible.

For staged workflows, upload the outputs from the prior stage in a structure that matches the later command's file discovery expectations. For example, a later analysis command may need a completed coding workbook with the configured filename.

## Use Examples Before Study Data

In Part 3, the examples download button changes with command selection:

- no selected commands: download the full example dataset;
- selected commands: download command-specific examples for those commands.

This is often the easiest way to see what the web app expects before uploading study data.

## Stage Manual Workflows

For manual coding workflows, use the web app one stage at a time:

1. generate coding files;
2. download the ZIP;
3. complete or review the coding workbook outside the app;
4. upload the completed workbook for evaluation, analysis, or rates.

If identifiers should be blinded for manual coding, encode before distributing coder-facing files. Decode back to original sample identifiers before DIAAD analysis when downstream merges require the original IDs. A project may re-encode final exports if blinded statistical analysis is preferred.

## Common Problems

If the app reports incomplete configuration, upload both split config files or finish the in-app builder and choose the built configuration.

If a later command cannot find a file, check the uploaded folder structure and configured filenames.

If a workflow needs a local dependency such as a spaCy model that the hosted app does not provide, use local web or CLI.

## Read Next

- Web app operation: `docs/manual/02_operation/03_webapp.md`
- Configured filenames and file discovery: `docs/manual/05_functionalities/09_configured_filenames_file_discovery_input_selection/02_usage_guide.md`
- Blinding workflow guidance: `docs/manual/05_functionalities/14_blinding_unblinding_auto_blind/02_usage_guide.md`
- Example package generation: `docs/manual/05_functionalities/04_example_package_generation_manifests/02_usage_guide.md`
