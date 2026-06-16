# Example Dataset and Command-Specific Packages Quickstart

Use generated examples before running an unfamiliar DIAAD command on study data. The examples are synthetic toy files, not participant data.

## Full Example Dataset

Generate the full example dataset:

```bash
diaad examples
```

This creates a comprehensive atlas of DIAAD inputs and expected outputs:

```text
example_files_full_dataset/
```

Use it when you want to browse the overall DIAAD file landscape.

## Command-Specific Examples

Generate a smaller package for one command:

```bash
diaad examples --for-command "transcripts tabularize"
```

Generate one package for a small command set by repeating the flag:

```bash
diaad examples --for-command "cus analyze" --for-command "cus evaluate"
```

Command-specific packages use `example_` prefixes:

```text
example_config/
example_input/
example_output/
example_logs/
```

Those packages more closely resemble a tangible toy project for the requested command or command set.

## Web App Examples

In the web app, use the example download action in Part 3:

- select no commands to download the full example dataset;
- select commands to download command-specific examples for those functions.

## Manual Example I/O

Generated Example I/O pages in the manual show readable snippets and previews. Downloadable example packages are hands-on file sets. They support the same learning goal, but they do not need to match exactly.

## Read Next

- Examples module: `docs/manual/04_modules/09_examples/01_quickstart.md`
- Examples command: `docs/manual/04_modules/09_examples/05_commands/01_examples/01_quickstart.md`
- Generated Example I/O feature: `docs/manual/03_features/04_generated_example_io.md`
- CLI setup workflow: `docs/manual/06_workflows/01_cli_project_setup_first_run/01_quickstart.md`
