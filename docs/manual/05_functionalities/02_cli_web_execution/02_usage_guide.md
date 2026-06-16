# CLI and Web Execution Usage Guide

The CLI and web app expose the same DIAAD work, but they fit different project situations.

## Choose CLI When

Use the CLI when:

- the data are identifiable, sensitive, or hard to deidentify;
- a workflow will be repeated across sites, batches, or timepoints;
- command-line overrides or scripted runs are needed;
- outputs need to stay in a controlled local directory;
- a workflow depends on inspecting intermediate files before continuing;
- a command needs local dependencies such as a specific spaCy model;
- detailed run artifacts are needed for audit and reproducibility.

CLI output is written under the configured output directory in a timestamped `diaad_YYMMDD_HHMM/` folder.

## Choose Web App When

Use the web app when:

- learning DIAAD with generated examples;
- testing a small deidentified dataset;
- building configuration interactively;
- downloading one output ZIP is easier than managing local Python commands;
- a teaching or demonstration context benefits from menus.

The web app writes files into a temporary workspace and returns a ZIP. Download the ZIP before closing or refreshing the session.

## Command Selection

The canonical ordinary command names come from `MODULE_COMMANDS` in `src/diaad/cli/commands.py`. The dispatch table in `src/diaad/cli/dispatch.py` maps supported command strings to run wrappers.

In the CLI, users type command strings:

```bash
diaad "transcripts tabularize, templates samples"
```

In the web app, users select commands from a module-organized checkbox menu. The examples action is separate: if no commands are selected, the examples button downloads the full example dataset; if commands are selected, it downloads selected-command examples.

## Chained Runs

Chained CLI runs can be convenient when one command creates an input needed by a later command:

```bash
diaad "transcripts tabularize, cus files"
```

Use chaining cautiously for workflows that require human review between steps. For example, manual coding workflows normally require users to generate a coding file, complete or review it, and only then run evaluation or analysis.

## Inputs And Outputs

CLI runs use configured local paths. Relative paths are resolved from the project root, normally the directory where the command is run.

Web runs replace configured input and output paths with temporary web-session folders. Uploaded folder structure is preserved under the temporary input directory when possible, and outputs are returned as a ZIP.

## Practical CLI-First Cases

Some ordinary commands are exposed through both routes but remain practically CLI-first for many projects:

- workflows with sensitive transcript content;
- workflows that need careful staged upload and download of manually edited workbooks;
- workflows with large folder trees;
- workflows that rely on local dependency installation;
- workflows that need complete CLI provenance artifacts.

This is a practical recommendation, not necessarily a code limitation.

## Common Problems

If the CLI rejects a command, compare it with the canonical command table in Command-Line Operation (`docs/manual/02_operation/02_command_line.md`).

If the web app cannot find files for a later-stage command, upload the folder from the earlier stage with its output structure preserved.

If `diaad streamlit` fails because Streamlit is missing, install the web extra:

```bash
pip install "diaad[web]"
```

If an examples-specific flag is used with an ordinary command or with `diaad streamlit`, DIAAD raises an error. Use `--for-command`, `--force`, and `--render-docs` only with `diaad examples`.

## Read Next

- Command-line operation: `docs/manual/02_operation/02_command_line.md`
- Web app operation: `docs/manual/02_operation/03_webapp.md`
- Examples module: `docs/manual/04_modules/09_examples/01_quickstart.md`
- Generated Example I/O: `docs/manual/03_features/04_generated_example_io.md`
